/*
Copyright (c) 2013, Michael Ossmann <mike@ossmann.com>
Copyright (c) 2012, Jared Boone <jared@sharebrained.com>
Copyright (c) 2014, Youssef Touil <youssef@airspy.com>
Copyright (c) 2014, Benjamin Vernoux <bvernoux@airspy.com>

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

		Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
		Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the 
		documentation and/or other materials provided with the distribution.
		Neither the name of Great Scott Gadgets nor the names of its contributors may be used to endorse or promote products derived from this software
		without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdlib.h>
#include <string.h>
#include <libusb.h>
#include <pthread.h>

#include "airspy.h"
#include "airspy_commands.h"
#include "iqconverter_float.h"
#include "iqconverter_int16.h"
#include "filters.h"

#ifndef bool
typedef int bool;
#define true 1
#define false 0
#endif

#define USE_PACKING false
#define PACKET_SIZE (12)
#define UNPACKED_SIZE (16)

#ifdef AIRSPY_BIG_ENDIAN
#define TO_LE(x) __builtin_bswap32(x)
#else
#define TO_LE(x) x
#endif

#define SAMPLE_SHIFT 3
#define SAMPLE_SCALE (1.0f / (1 << (15 - SAMPLE_SHIFT)))
#define SAMPLE_RESOLUTION 12

typedef struct {
	uint32_t freq_hz;
} set_freq_params_t;

typedef struct airspy_device
{
	libusb_device_handle* usb_device;
	struct libusb_transfer** transfers;
	airspy_sample_block_cb_fn callback;
	volatile bool streaming;
	volatile bool stop_requested;
	volatile bool data_available;
	pthread_t transfer_thread;
	pthread_t conversion_thread;
	pthread_cond_t conversion_cv;
	pthread_mutex_t conversion_mp;
	uint32_t transfer_count;
	uint32_t buffer_size;
	uint32_t total_dropped_samples;
	unsigned char *received_buffer;
	uint16_t *raw_samples;
	void *output_buffer;
	iqconveter_float_t *cnv_f;
	iqconveter_int16_t *cnv_i;
	void* ctx;
	enum airspy_sample_type sample_type;
} airspy_device_t;

static const uint16_t airspy_usb_vid = 0x1d50;
static const uint16_t airspy_usb_pid = 0x60a1;

#define USB_PRODUCT_ID (2)
#define STR_DESCRIPTOR_SIZE (250)
unsigned char str_desc[STR_DESCRIPTOR_SIZE+1] = { 0 };

#define STR_PRODUCT_AIRSPY_SIZE (6)
const unsigned char str_product_airspy[STR_PRODUCT_AIRSPY_SIZE] = { 'A', 'I', 'R', 'S', 'P', 'Y' };

static libusb_context* g_libusb_context = NULL;

static int cancel_transfers(airspy_device_t* device)
{
	uint32_t transfer_index;

	if( device->transfers != NULL )
	{
		for(transfer_index=0; transfer_index<device->transfer_count; transfer_index++)
		{
			if( device->transfers[transfer_index] != NULL )
			{
				libusb_cancel_transfer(device->transfers[transfer_index]);
			}
		}
		return AIRSPY_SUCCESS;
	} else {
		return AIRSPY_ERROR_OTHER;
	}
}

static int free_transfers(airspy_device_t* device)
{
	uint32_t transfer_index;

	if (device->transfers != NULL)
	{
		// libusb_close() should free all transfers referenced from this array.
		for(transfer_index=0; transfer_index < device->transfer_count; transfer_index++)
		{
			if( device->transfers[transfer_index] != NULL )
			{
				libusb_free_transfer(device->transfers[transfer_index]);
				device->transfers[transfer_index] = NULL;
			}
		}
		free(device->transfers);
		device->transfers = NULL;

		free(device->output_buffer);
		free(device->received_buffer);
		free(device->raw_samples);
	}

	return AIRSPY_SUCCESS;
}

static int allocate_transfers(airspy_device_t* const device)
{
	size_t sample_count;
	uint32_t transfer_index;

	if( device->transfers == NULL )
	{
		device->received_buffer = (unsigned char *) malloc(device->buffer_size);
		if (device->received_buffer == NULL)
		{
			return AIRSPY_ERROR_NO_MEM;
		}

#if (USE_PACKING)
		sample_count = device->buffer_size / 2 * UNPACKED_SIZE / PACKET_SIZE;
#else
		sample_count = device->buffer_size / 2;
#endif

		device->raw_samples = (uint16_t *) malloc(sample_count * sizeof(uint16_t));
		if (device->raw_samples == NULL)
		{
			return AIRSPY_ERROR_NO_MEM;
		}

		device->output_buffer = (float *) malloc(sample_count * sizeof(float));
		if (device->output_buffer == NULL)
		{
			return AIRSPY_ERROR_NO_MEM;
		}

		device->transfers = (struct libusb_transfer**) calloc(device->transfer_count, sizeof(struct libusb_transfer));
		if( device->transfers == NULL )
		{
			return AIRSPY_ERROR_NO_MEM;
		}

		for(transfer_index=0; transfer_index<device->transfer_count; transfer_index++)
		{
			device->transfers[transfer_index] = libusb_alloc_transfer(0);
			if( device->transfers[transfer_index] == NULL )
			{
				return AIRSPY_ERROR_LIBUSB;
			}

			libusb_fill_bulk_transfer(
				device->transfers[transfer_index],
				device->usb_device,
				0,
				(unsigned char*)malloc(device->buffer_size),
				device->buffer_size,
				NULL,
				device,
				0
			);

			if( device->transfers[transfer_index]->buffer == NULL )
			{
				return AIRSPY_ERROR_NO_MEM;
			}
		}
		return AIRSPY_SUCCESS;
	}
	else
	{
		return AIRSPY_ERROR_BUSY;
	}
}

static int prepare_transfers(airspy_device_t* device, const uint_fast8_t endpoint_address, libusb_transfer_cb_fn callback)
{
	int error;
	uint32_t transfer_index;
	if( device->transfers != NULL )
	{
		for(transfer_index=0; transfer_index<device->transfer_count; transfer_index++)
		{
			device->transfers[transfer_index]->endpoint = endpoint_address;
			device->transfers[transfer_index]->callback = callback;

			error = libusb_submit_transfer(device->transfers[transfer_index]);
			if( error != 0 )
			{
				return AIRSPY_ERROR_LIBUSB;
			}
		}
		return AIRSPY_SUCCESS;
	} else {
		// This shouldn't happen.
		return AIRSPY_ERROR_OTHER;
	}
}

static void convert_samples_int16(uint16_t *src, int16_t *dest, int count)
{
	int i;
	for (i = 0; i < count; i++)
	{
		dest[i] = (src[i] - 2048) << SAMPLE_SHIFT;
	}
}

static void convert_samples_float(uint16_t *src, float *dest, int count)
{
	int i;
	for (i = 0; i < count; i++)
	{
		dest[i] = (src[i] - 2048) * SAMPLE_SCALE;
	}
}

#if (USE_PACKING)

static void unpack_samples(unsigned char *src, uint16_t *dest, int len)
{
	int i;
	int iter;
	uint32_t *packed_data;

	iter = len / PACKET_SIZE;
	packed_data = (uint32_t *) src;

	for (i = 0; i < iter; i++)
	{
		/*dest[0] = packed_data[0] >> 20;
		dest[1] = (packed_data[0] << 12) >> 20;
		dest[2] = ((packed_data[0] & 0xFF) << 4) | (packed_data[1] >> 28);
		dest[3] = (packed_data[1] >> 16) & 0xFFF;
		dest[4] = (packed_data[1] >> 4) & 0xFFF;
		dest[5] = ((packed_data[1] & 0xF) << 8) | (packed_data[2] >> 24);
		dest[6] = (packed_data[2] >> 12) & 0xFFF;
		dest[7] = packed_data[2] & 0xFFF;*/

		dest[0] = packed_data[0] & 0xFFF;
		dest[1] = (packed_data[0] >> 12) & 0xFFF;
		dest[2] = (packed_data[0] >> 24) | ((packed_data[1] << 8) & 0xF00);
		dest[3] = (packed_data[1] >> 4) & 0xFFF;
		dest[4] = (packed_data[1] >> 16) & 0xFFF;
		dest[5] = (packed_data[1] >> 28) | ((packed_data[2] << 4) & 0xFF0);
		dest[6] = (packed_data[2] >> 8) & 0xFFF;
		dest[7] = packed_data[2] >> 20;

		dest += 8;
		packed_data += 3;
	}
}

#endif

static void* conversion_threadproc(void *arg)
{
	int sample_count;
	airspy_device_t* device = (airspy_device_t*)arg;
	airspy_transfer_t transfer;

#ifdef _WIN32

	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

#endif

	while (device->streaming && !device->stop_requested)
	{

#if (USE_PACKING)
		unpack_samples(device->received_buffer, device->raw_samples, device->buffer_size);
		sample_count = device->buffer_size / 2 * UNPACKED_SIZE / PACKET_SIZE;
#else
		memcpy(device->raw_samples, device->received_buffer, device->buffer_size);
		sample_count = device->buffer_size / 2;
#endif

		device->data_available = false;

		switch (device->sample_type)
		{
		case AIRSPY_SAMPLE_FLOAT32_IQ:
			convert_samples_float(device->raw_samples, (float *) device->output_buffer, sample_count);
			iqconverter_float_process(device->cnv_f, (float *) device->output_buffer, sample_count);
			sample_count /= 2;
			break;

		case AIRSPY_SAMPLE_FLOAT32_REAL:
			convert_samples_float(device->raw_samples, (float *) device->output_buffer, sample_count);
			break;

		case AIRSPY_SAMPLE_INT16_IQ:
			convert_samples_int16(device->raw_samples, (int16_t *) device->output_buffer, sample_count);
			iqconverter_int16_process(device->cnv_i, (int16_t *) device->output_buffer, sample_count);
			sample_count /= 2;
			break;

		case AIRSPY_SAMPLE_INT16_REAL:
			convert_samples_int16(device->raw_samples, (int16_t *) device->output_buffer, sample_count);
			break;
		}

		transfer.device = device;
		transfer.ctx = device->ctx;
		transfer.samples = device->output_buffer;
		transfer.sample_count = sample_count;
		transfer.sample_type = device->sample_type;

		if (device->callback(&transfer) != 0)
		{
			device->stop_requested = true;
		}

		pthread_mutex_lock(&device->conversion_mp);
		while (!device->data_available && !device->stop_requested && device->streaming)
		{
			pthread_cond_wait(&device->conversion_cv, &device->conversion_mp);
		}
		pthread_mutex_unlock(&device->conversion_mp);
	}

	return NULL;
}

static void airspy_libusb_transfer_callback(struct libusb_transfer* usb_transfer)
{
	airspy_device_t* device = (airspy_device_t*) usb_transfer->user_data;

	if (!device->streaming || device->stop_requested)
	{
		return;
	}

	if (usb_transfer->status != LIBUSB_TRANSFER_COMPLETED)
	{
		device->streaming = false;
		return;
	}

	if (!device->data_available)
	{
		memcpy(device->received_buffer, usb_transfer->buffer, usb_transfer->length);
		device->data_available = true;

		pthread_mutex_lock(&device->conversion_mp);
		pthread_cond_signal(&device->conversion_cv);
		pthread_mutex_unlock(&device->conversion_mp);
	}

	if (libusb_submit_transfer(usb_transfer) != 0)
	{
		device->streaming = false;
	}
}

static void* transfer_threadproc(void* arg)
{
	airspy_device_t* device = (airspy_device_t*)arg;
	int error;
	struct timeval timeout = { 0, 500000 };

#ifdef _WIN32

	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

#endif

	while (device->streaming && !device->stop_requested)
	{
		error = libusb_handle_events_timeout_completed(g_libusb_context, &timeout, NULL);
		if (error < 0)
		{
			if (error != LIBUSB_ERROR_INTERRUPTED)
				device->streaming = false;
		}
	}

	return NULL;
}

static int kill_io_threads(airspy_device_t* device)
{
	if (device->streaming)
	{
		device->stop_requested = true;
		cancel_transfers(device);

		pthread_cond_signal(&device->conversion_cv);

		pthread_join(device->transfer_thread, NULL);
		pthread_join(device->conversion_thread, NULL);

		device->stop_requested = false;
		device->streaming = false;
		device->data_available = false;
	}

	return AIRSPY_SUCCESS;
}

static int create_io_threads(airspy_device_t* device, airspy_sample_block_cb_fn callback)
{
	int result;
	pthread_attr_t attr;

	if (!device->streaming && !device->stop_requested)
	{
		device->callback = callback;
		device->streaming = true;

		result = prepare_transfers(device, LIBUSB_ENDPOINT_IN | 1, (libusb_transfer_cb_fn) airspy_libusb_transfer_callback);
		if (result != AIRSPY_SUCCESS)
		{
			return result;
		}

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		result = pthread_create(&device->conversion_thread, &attr, conversion_threadproc, device);
		if (result != 0)
		{
			return AIRSPY_ERROR_THREAD;
		}

		result = pthread_create(&device->transfer_thread, &attr, transfer_threadproc, device);
		if (result != 0)
		{
			return AIRSPY_ERROR_THREAD;
		}

		pthread_attr_destroy(&attr);
	}
	else {
		return AIRSPY_ERROR_BUSY;
	}

	return AIRSPY_SUCCESS;
}

#ifdef __cplusplus
extern "C"
{
#endif

int ADDCALL airspy_init(void)
{
	int i;
	const int libusb_error = libusb_init(&g_libusb_context);

	if( libusb_error != 0 )
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_exit(void)
{
	if( g_libusb_context != NULL )
	{
		libusb_exit(g_libusb_context);
		g_libusb_context = NULL;
	}

	return AIRSPY_SUCCESS;
}

int ADDCALL airspy_open(airspy_device_t** device)
{
	int result;
	libusb_device_handle* usb_device;
	airspy_device_t* lib_device;

	if( device == NULL )
	{
		return AIRSPY_ERROR_INVALID_PARAM;
	}
	// TODO: Do proper scanning of available devices, searching for
	// unit serial number (if specified?).
	usb_device = libusb_open_device_with_vid_pid(g_libusb_context, airspy_usb_vid, airspy_usb_pid);
	if( usb_device == NULL )
	{
		return AIRSPY_ERROR_NOT_FOUND;
	}

	/* Get Product Descriptor */
	result = libusb_get_string_descriptor_ascii(usb_device, USB_PRODUCT_ID, str_desc, STR_DESCRIPTOR_SIZE);
	if( result != 0 )
	{
		/* Check Product corresponds to AIRSPY product */
		result = memcmp(str_desc, str_product_airspy, STR_PRODUCT_AIRSPY_SIZE);
		if(result != 0)
		{
			libusb_close(usb_device);
			return AIRSPY_ERROR_NOT_FOUND;
		}
	}else
	{
		libusb_close(usb_device);
		return AIRSPY_ERROR_LIBUSB;
	}

	result = libusb_set_configuration(usb_device, 1);
	if( result != 0 )
	{
		libusb_close(usb_device);
		return AIRSPY_ERROR_LIBUSB;
	}

	result = libusb_claim_interface(usb_device, 0);
	if( result != 0 )
	{
		libusb_close(usb_device);
		return AIRSPY_ERROR_LIBUSB;
	}

	lib_device = NULL;
	lib_device = (airspy_device_t*)malloc(sizeof(*lib_device));
	if( lib_device == NULL )
	{
		libusb_release_interface(usb_device, 0);
		libusb_close(usb_device);
		return AIRSPY_ERROR_NO_MEM;
	}

	lib_device->usb_device = usb_device;
	lib_device->transfers = NULL;
	lib_device->callback = NULL;
	/*
	lib_device->transfer_count = 1024;
	lib_device->buffer_size = 16384;
	*/
	lib_device->transfer_count = 10;
	//lib_device->buffer_size = 262140; /* Must be a multiple of 12 does not work with linux */
	lib_device->buffer_size = 262144; /* Work with linux */
	lib_device->streaming = false;
	lib_device->stop_requested = false;
	lib_device->data_available = false;
	lib_device->sample_type = AIRSPY_SAMPLE_FLOAT32_IQ;

	result = allocate_transfers(lib_device);
	if( result != 0 )
	{
		free(lib_device);
		libusb_release_interface(usb_device, 0);
		libusb_close(usb_device);
		return AIRSPY_ERROR_NO_MEM;
	}

	lib_device->cnv_f = iqconverter_float_create(HB_KERNEL_FLOAT, HB_KERNEL_FLOAT_LEN);
	lib_device->cnv_i = iqconverter_int16_create(HB_KERNEL_INT16, HB_KERNEL_INT16_LEN);

	pthread_cond_init(&lib_device->conversion_cv, NULL);
	pthread_mutex_init(&lib_device->conversion_mp, NULL);

	*device = lib_device;

	return AIRSPY_SUCCESS;
}

int ADDCALL airspy_close(airspy_device_t* device)
{
	int result;

	result = AIRSPY_SUCCESS;
	
	if (device != NULL)
	{
		result = airspy_stop_rx(device);

		if( device->usb_device != NULL )
		{
			libusb_release_interface(device->usb_device, 0);
			libusb_close(device->usb_device);
			device->usb_device = NULL;
		}

		free_transfers(device);

		iqconverter_float_free(device->cnv_f);
		iqconverter_int16_free(device->cnv_i);

		pthread_cond_destroy(&device->conversion_cv);
		pthread_mutex_destroy(&device->conversion_mp);

		free(device);
	}

	return result;
}

int ADDCALL airspy_set_receiver_mode(airspy_device_t* device, receiver_mode_t value)
{
	int result;
	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_RECEIVER_MODE,
		value,
		0,
		NULL,
		0,
		0
	);

	if( result != 0 )
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_start_rx(airspy_device_t* device, airspy_sample_block_cb_fn callback, void* ctx)
{
	int result;

	result = airspy_set_receiver_mode(device, RECEIVER_MODE_RX);
	if( result == AIRSPY_SUCCESS )
	{
		device->ctx = ctx;
		result = create_io_threads(device, callback);
	}
	return result;
}

int ADDCALL airspy_stop_rx(airspy_device_t* device)
{
	int result1, result2;
	result1 = kill_io_threads(device);

	result2 = airspy_set_receiver_mode(device, RECEIVER_MODE_OFF);
	if (result2 != AIRSPY_SUCCESS)
	{
		return result2;
	}
	return result1;
}

int ADDCALL airspy_si5351c_read(airspy_device_t* device, uint8_t register_number, uint8_t* value)
{
	uint8_t temp_value;
	int result;

	temp_value = 0;
	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SI5351C_READ,
		0,
		register_number,
		(unsigned char*)&temp_value,
		1,
		0);

	if( result < 1 )
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		*value = temp_value;
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_si5351c_write(airspy_device_t* device, uint8_t register_number, uint8_t value)
{
	int result;

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SI5351C_WRITE,
		value,
		register_number,
		NULL,
		0,
		0);

	if( result != 0 )
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_r820t_read(airspy_device_t* device, uint8_t register_number, uint8_t* value)
{
	int result;

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_R820T_READ,
		0,
		register_number,
		(unsigned char*) value,
		1,
		0);

	if( result < 1 )
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_r820t_write(airspy_device_t* device, uint8_t register_number, uint8_t value)
{
	int result;
	
	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_R820T_WRITE,
		value,
		register_number,
		NULL,
		0,
		0);

	if( result != 0 )
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_spiflash_erase(airspy_device_t* device)
{
	int result;
	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SPIFLASH_ERASE,
		0,
		0,
		NULL,
		0,
		0);

	if (result != 0)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_spiflash_write(airspy_device_t* device, const uint32_t address, const uint16_t length, unsigned char* const data)
{
	int result;
	
	if (address > 0x0FFFFF)
	{
		return AIRSPY_ERROR_INVALID_PARAM;
	}

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SPIFLASH_WRITE,
		address >> 16,
		address & 0xFFFF,
		data,
		length,
		0);

	if (result < length)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_spiflash_read(airspy_device_t* device, const uint32_t address, const uint16_t length, unsigned char* data)
{
	int result;

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SPIFLASH_READ,
		address >> 16,
		address & 0xFFFF,
		data,
		length,
		0);

	if (result < length)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_board_id_read(airspy_device_t* device, uint8_t* value)
{
	int result;
	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_BOARD_ID_READ,
		0,
		0,
		value,
		1,
		0);

	if (result < 1)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_version_string_read(airspy_device_t* device, char* version, uint8_t length)
{
	int result;

	memset(version, 0, length);

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_VERSION_STRING_READ,
		0,
		0,
		(unsigned char*)version,
		(length-1),
		0);

	if (result < 0)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else
	{
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_board_partid_serialno_read(airspy_device_t* device, airspy_read_partid_serialno_t* read_partid_serialno)
{
	uint8_t length;
	int result;
	
	length = sizeof(airspy_read_partid_serialno_t);
	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_BOARD_PARTID_SERIALNO_READ,
		0,
		0,
		(unsigned char*)read_partid_serialno,
		length,
		0);

	if (result < length)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {

		read_partid_serialno->part_id[0] = TO_LE(read_partid_serialno->part_id[0]);
		read_partid_serialno->part_id[1] = TO_LE(read_partid_serialno->part_id[1]);
		read_partid_serialno->serial_no[0] = TO_LE(read_partid_serialno->serial_no[0]);
		read_partid_serialno->serial_no[1] = TO_LE(read_partid_serialno->serial_no[1]);
		read_partid_serialno->serial_no[2] = TO_LE(read_partid_serialno->serial_no[2]);
		read_partid_serialno->serial_no[3] = TO_LE(read_partid_serialno->serial_no[3]);

		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_set_sample_type(struct airspy_device* device, enum airspy_sample_type sample_type)
{
	device->sample_type = sample_type;
	return AIRSPY_SUCCESS;
}

int ADDCALL airspy_set_freq(airspy_device_t* device, const uint32_t freq_hz)
{
	set_freq_params_t set_freq_params;
	uint8_t length;
	int result;

	set_freq_params.freq_hz = TO_LE(freq_hz);
	length = sizeof(set_freq_params_t);

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SET_FREQ,
		0,
		0,
		(unsigned char*)&set_freq_params,
		length,
		0
	);

	if (result < length)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_set_lna_gain(airspy_device_t* device, uint8_t value)
{
	int result;
	uint8_t retval;
	uint8_t length;

	length = 1;

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SET_LNA_GAIN,
		0,
		value,
		&retval,
		length,
		0
	);

	if (result < length)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_set_mixer_gain(airspy_device_t* device, uint8_t value)
{
	int result;
	uint8_t retval;
	uint8_t length;

	length = 1;

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SET_MIXER_GAIN,
		0,
		value,
		&retval,
		length,
		0
	);

	if (result < length)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_set_vga_gain(airspy_device_t* device, uint8_t value)
{
	int result;
	uint8_t retval;
	uint8_t length;

	length = 1;

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SET_VGA_GAIN,
		0,
		value,
		&retval,
		length,
		0
	);

	if (result < length)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_set_lna_agc(airspy_device_t* device, uint8_t value)
{
	int result;
	uint8_t retval;
	uint8_t length;

	length = 1;

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SET_LNA_AGC,
		0,
		value,
		&retval,
		length,
		0
	);

	if (result < length)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_set_mixer_agc(airspy_device_t* device, uint8_t value)
{
	int result;
	uint8_t retval;
	uint8_t length;

	length = 1;

	result = libusb_control_transfer(
		device->usb_device,
		LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
		AIRSPY_SET_MIXER_AGC,
		0,
		value,
		&retval,
		length,
		0
	);

	if (result < length)
	{
		return AIRSPY_ERROR_LIBUSB;
	} else {
		return AIRSPY_SUCCESS;
	}
}

int ADDCALL airspy_is_streaming(airspy_device_t* device)
{
	return device->streaming == true;
}

const char* ADDCALL airspy_error_name(enum airspy_error errcode)
{
	switch(errcode)
	{
	case AIRSPY_SUCCESS:
		return "AIRSPY_SUCCESS";

	case AIRSPY_TRUE:
		return "AIRSPY_TRUE";

	case AIRSPY_ERROR_INVALID_PARAM:
		return "AIRSPY_ERROR_INVALID_PARAM";

	case AIRSPY_ERROR_NOT_FOUND:
		return "AIRSPY_ERROR_NOT_FOUND";

	case AIRSPY_ERROR_BUSY:
		return "AIRSPY_ERROR_BUSY";

	case AIRSPY_ERROR_NO_MEM:
		return "AIRSPY_ERROR_NO_MEM";

	case AIRSPY_ERROR_LIBUSB:
		return "AIRSPY_ERROR_LIBUSB";

	case AIRSPY_ERROR_THREAD:
		return "AIRSPY_ERROR_THREAD";

	case AIRSPY_ERROR_STREAMING_THREAD_ERR:
		return "AIRSPY_ERROR_STREAMING_THREAD_ERR";

	case AIRSPY_ERROR_STREAMING_STOPPED:
		return "AIRSPY_ERROR_STREAMING_STOPPED";

	case AIRSPY_ERROR_OTHER:
		return "AIRSPY_ERROR_OTHER";

	default:
		return "airspy unknown error";
	}
}

const char* ADDCALL airspy_board_id_name(enum airspy_board_id board_id)
{
	switch(board_id)
	{
	case AIRSPY_BOARD_ID_PROTO_AIRSPY:
		return "AIRSPY";

	case AIRSPY_BOARD_ID_INVALID:
		return "Invalid Board ID";

	default:
		return "Unknown Board ID";
	}
}

#ifdef __cplusplus
} // __cplusplus defined.
#endif
