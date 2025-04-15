"""
To install:
    pip install hidapi
    https://trezor.github.io/cython-hidapi/examples.html

NOTES:
    Spectrometer is little endian
"""
from __future__ import annotations

import logging
import math
import struct
import time
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from types import TracebackType
from typing import Optional, Type

import hid
import numpy as np

LOGGER = logging.getLogger(__name__)

VENDOR_ID = 0xE220  # 57888
PRODUCT_ID = 0x100  # 256
ENCODING = "utf-8"

ZERO_REPORT_ID = 0
PACKET_SIZE_BYTES = 64
STANDARD_TIMEOUT_MS = 100
PARAMETER_SET_DELAY_S = 0.1
MAX_PACKETS_IN_FRAME = 124
MAX_SPECTRA_MEMORIES = 64
REMAINING_PACKETS_ERROR = 250
NUM_OF_PIXELS_IN_PACKET = 30
FLASH_ERASE_TIMEOUT_MS = 5000
FLASH_MAX_READ_PACKETS = 100
FLASH_MAX_WRITE_BYTES = 58
FLASH_MAX_OFFSET = 0x1FFFF
FLASH_MAX_BYTES = 0x20000
CALIBRATION_LINES = 10976


class RequestCode(IntEnum):
    status = 1
    set_exposure = 2
    set_acquisition_parameters = 3
    set_frame_format = 4
    set_external_trigger = 5
    set_software_trigger = 6
    clear_memory = 7
    get_frame_format = 8
    get_acquisition_parameters = 9
    set_all_parameters = 0x0C
    get_frame = 0x0A
    set_optical_trigger = 0x0B
    read_flash = 0x1A
    write_flash = 0x1B
    erase_flash = 0x1C
    reset = 0xF1
    detach = 0xF2


class ReplyCode(IntEnum):
    status = 0x81
    set_exposure = 0x82
    set_acquisition_parameters = 0x83
    set_frame_format = 0x84
    set_external_trigger = 0x85
    set_software_trigger = 0x86
    clear_memory = 0x87
    get_frame_format = 0x88
    get_acquisition_parameters = 0x89
    set_all_parameters = 0x8C
    get_frame = 0x8A
    set_optical_trigger = 0x8B
    read_flash = 0x9A
    write_flash = 0x9B
    erase_flash = 0x9C


class TriggerMode(IntEnum):
    disabled = 0
    enabled = 1
    oneshot = 2


class TriggerSlope(IntEnum):
    disabled = 0
    rising = 1
    falling = 2
    rise_fall = 3


class ScanMode(IntEnum):
    continuous = 0
    idle = 1
    every_frame_idle = 2
    frame_averaging = 3


class AverageMode(IntEnum):
    disabled = 0
    average_2 = 1
    average_4 = 2
    average_8 = 3


class Status(IntFlag):
    idle = 0
    in_progress = 1
    memory_full = 2


@dataclass
class Parameters:
    scan_count: int = 1
    blank_scan_count: int = 0
    scan_mode: ScanMode = ScanMode.continuous
    exposure_time_ms: int = 10

    def from_bytes(self, report: list) -> Self:
        # Assumes the incoming report still has the first ID byte.
        (
            self.scan_count,
            self.blank_scan_count,
            scan_mode,
            exp_10s_of_us,
        ) = struct.unpack("<HHBL", bytearray(report[1:10]))

        self.scan_mode = ScanMode(scan_mode)
        self.exposure_time_ms = exp_10s_of_us / 100
        return self

    def to_bytes(self) -> bytearray:
        exp_10s_of_us = int(self.exposure_time_ms * 100)
        report = struct.pack(
            "<HHBL",
            self.scan_count,
            self.blank_scan_count,
            self.scan_mode.value,
            exp_10s_of_us,
        )
        return report


@dataclass
class FrameFormat:
    start_element: int = 1
    end_element: int = 10
    reduction_mode: AverageMode = AverageMode.disabled
    pixels_in_frame: int = 10

    def from_bytes(self, report: list) -> Self:
        # Assumes the incoming report still has the first ID byte.
        (
            self.start_element,
            self.end_element,
            reduction_mode,
            self.pixels_in_frame,
        ) = struct.unpack("<HHBH", bytearray(report[1:8]))
        self.reduction_mode = AverageMode(reduction_mode)
        return self

    def to_bytes(self) -> bytearray:
        report = struct.pack(
            "<HHBH",
            self.start_element,
            self.end_element,
            self.reduction_mode.value,
            self.pixels_in_frame,
        )
        return report


@dataclass
class Calibration:
    """
    - Calibration is an ASCII file
    - Only the c.Y type with irradiance calibration is supported.
    - Detector has 3648, thus offsets are needed
        - Wave array has 3653 elements
        - prnu and irr array has 3654 elements
    - Blank memory locations are stored with 0xFF so those are filtered from the
      end of the input array.
    """

    model: str = None
    type: str = None
    serial: int = None
    irr_scaler: float = None
    irr_wave: float = None
    _wavelengths: np.array = np.ones(3653)
    _prnu_norm: np.array = np.ones(3654)
    _irr_norm: np.array = np.ones(3654)

    @property
    def wavelengths(self):
        return self._wavelengths[:-5]

    @property
    def prnu_norm(self):
        return self._prnu_norm[:-6]

    @property
    def irr_norm(self):
        return self._irr_norm[:-6]

    def from_bytes(self, raw: bytearray) -> Self:
        while raw and raw[-1] == 0xFF:
            raw.pop()
        lines = raw.decode(ENCODING).replace("\t", "").replace("\r", "").split("\n")
        if len(lines) != CALIBRATION_LINES:
            raise ValueError(
                f"Invalid calibration length.  Expected {CALIBRATION_LINES} lines, got {len(lines)}"
            )
        header = lines[0].split()
        self.model = header[0]
        self.type = header[1]
        self.serial = int(header[2])
        self.irr_scaler = float(lines[1])
        self.irr_wave = float(lines[2])
        self._wavelengths = np.asarray(lines[12:3665]).astype(float)
        self._prnu_norm = np.asarray(lines[3666:7320]).astype(float)
        self._irr_norm = np.asarray(lines[7321:10975]).astype(float)
        return self

    def to_bytes(self) -> bytearray:
        report = [""] * CALIBRATION_LINES
        report[0] = f"{self.model} {self.type} {self.serial}"
        report[1] = f"{self.irr_scaler:.6e}"
        report[2] = f"{self.irr_wave:.6f}"
        report[12:3665] = self._wavelengths.astype(str)
        report[3666:7320] = self._prnu_norm.astype(str)
        report[7321:10975] = self._irr_norm.astype(str)
        report = "\n".join(report)
        return bytearray(report, encoding=ENCODING)

    def from_file(self, file_path: str) -> Self:
        with open(file_path, "rb") as f:
            self.from_bytes(f.read())

    def to_file(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            f.write(self.to_bytes())


class LR1:
    @classmethod
    def discover(self, target_serial_no: str = None) -> LR1:
        for device_dict in enumerate():
            serial_no = device_dict["serial_number"]

            if target_serial_no:
                if serial_no == target_serial_no:
                    return LR1(target_serial_no)
            else:
                # Grab the first one
                return LR1(serial_no)

        raise OSError("No LR1 spectrometers found. Check connection and device power.")

    def __init__(self, serial_no: Optional[str] = None) -> None:
        self.serial_no = serial_no
        self.device = hid.device()
        self.connected = False
        self.status = None
        self.frames_in_mem = None
        self.parameters = None
        self.frame_format = None
        self.calibration = None

    def __str__(self) -> str:
        return f"Spectrometer [{self.serial_no or 'ASQ_SPC???????'}]: {'' if self.connected else 'dis'}connected"

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self.close()
        return False

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value
        if self._verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def open(self) -> None:
        try:
            self.device.open(
                VENDOR_ID,
                PRODUCT_ID,
                self.serial_no,
            )
            self.reset()
        except OSError:
            raise OSError(f"Unable to open spectrometer serial no: {self.serial_no}")

        LOGGER.debug(f"Connected to {self.serial_no}")
        self.connected = True
        self.get_parameters()
        self.get_frame_format()
        self.get_status()
        self.get_calibration()

    def close(self) -> None:
        self.device.close()
        self.connected = False
        self.status = None
        self.frames_in_mem = None
        self.parameters = None
        self.frame_format = None
        self.calibration = None
        LOGGER.debug(f"Device Closed")

    def receive(self, correct_reply: ReplyCode, timeout_ms: int) -> list:
        reply = None
        start_time = time.time()
        current_time = start_time
        reply = self.device.read(PACKET_SIZE_BYTES, timeout_ms)
        if reply[0] == correct_reply.value:
            return reply
        else:
            raise OSError(f"Incorrect reply {reply}")

    def send(self, report: bytes) -> None:
        try:
            self.device.write(report)
        except Exception as e:
            raise OSError(f"Unable to write to device.\n{e}")

    def send_and_receive(
        self,
        report: bytes,
        correct_reply: int,
        timeout_ms: int,
    ) -> list:
        self.send(report)
        results = self.receive(correct_reply, timeout_ms)
        return results

    def get_status(self) -> Status:
        report = [ZERO_REPORT_ID, RequestCode.status.value, 0x00]
        reply = self.send_and_receive(report, ReplyCode.status, STANDARD_TIMEOUT_MS)
        self.status = Status(reply[1])
        self.frames_in_mem = int.from_bytes(report[2:4], byteorder="little")
        return self.status

    def reset(self) -> None:
        report = [ZERO_REPORT_ID, RequestCode.reset.value]
        self.send(report)
        LOGGER.debug(f"Device Reset")

    def detach(self) -> None:
        report = [ZERO_REPORT_ID, RequestCode.detach.value]
        self.send(report)
        LOGGER.debug(f"Device Detached")

    def get_parameters(self) -> Parameters:
        LOGGER.debug("Loading Parameters")
        report = [
            ZERO_REPORT_ID,
            RequestCode.get_acquisition_parameters.value,
            0x00,
        ]
        reply = self.send_and_receive(
            report,
            ReplyCode.get_acquisition_parameters,
            STANDARD_TIMEOUT_MS,
        )
        self.parameters = Parameters().from_bytes(reply)
        return self.parameters

    def set_parameters(self) -> None:
        LOGGER.debug("Setting Parameters")
        report = [ZERO_REPORT_ID, RequestCode.set_acquisition_parameters.value]
        report += self.parameters.to_bytes()
        _ = self.send_and_receive(
            report,
            ReplyCode.set_acquisition_parameters,
            STANDARD_TIMEOUT_MS,
        )
        time.sleep(PARAMETER_SET_DELAY_S)  # wait for parameters to update

    def set_exposure_ms(self, exposure_ms: int) -> None:
        LOGGER.debug(f"Setting exposure to {exposure_ms} ms")
        self.parameters.exposure_time_ms = exposure_ms
        report = [ZERO_REPORT_ID, RequestCode.set_exposure.value]
        report += self.parameters.to_bytes()[-4:]
        _ = self.send_and_receive(
            report,
            ReplyCode.set_exposure,
            STANDARD_TIMEOUT_MS,
        )

    def get_frame_format(self) -> FrameFormat:
        LOGGER.debug("Getting Frame Format")
        report = [ZERO_REPORT_ID, RequestCode.get_frame_format.value]
        reply = self.send_and_receive(
            report,
            ReplyCode.get_frame_format,
            STANDARD_TIMEOUT_MS,
        )
        self.frame_format = FrameFormat().from_bytes(reply)
        return self.frame_format

    def set_frame_format(self) -> None:
        LOGGER.debug("Setting Frame Format")
        report = [ZERO_REPORT_ID, RequestCode.set_frame_format.value]
        report += self.frame_format.to_bytes()
        _ = self.send_and_receive(
            report,
            ReplyCode.set_frame_format,
            STANDARD_TIMEOUT_MS,
        )

    def set_external_trigger(self, mode: TriggerMode, slope: TriggerSlope) -> None:
        report = [
            ZERO_REPORT_ID,
            RequestCode.set_external_trigger.value,
            mode.value,
            slope.value,
        ]

        reply = self.send_and_receive(
            report,
            ReplyCode.set_external_trigger,
            STANDARD_TIMEOUT_MS,
        )

    def set_optical_trigger(
        self,
        mode: TriggerMode,
        pixel_index: int,
        threshold: int,
    ) -> None:
        report = struct.pack(
            "<BBHH",
            ZERO_REPORT_ID,
            RequestCode.set_optical_trigger.value,
            pixel_index,
            threshold,
        )
        _ = self.send_and_receive(
            report,
            ReplyCode.set_optical_trigger,
            STANDARD_TIMEOUT_MS,
        )

    def software_trigger(self) -> None:
        LOGGER.debug("Software Trigger")
        report = [ZERO_REPORT_ID, RequestCode.set_software_trigger.value]
        self.send(report)

    def clear_memory(self) -> None:
        LOGGER.debug(f"Clearing Memory")
        report = [ZERO_REPORT_ID, RequestCode.clear_memory.value]
        _ = self.send_and_receive(
            report,
            ReplyCode.clear_memory,
            STANDARD_TIMEOUT_MS,
        )

    def get_raw_frame(self, buffer_index: int = 0, offset: int = 0) -> list:
        if buffer_index >= MAX_SPECTRA_MEMORIES:
            raise ValueError(f"buffer_index limited to {MAX_SPECTRA_MEMORIES}")

        LOGGER.debug(f"Reading Frame from index {buffer_index}")
        pixels_in_frame = self.frame_format.pixels_in_frame
        packets_to_get = int(math.ceil(pixels_in_frame / NUM_OF_PIXELS_IN_PACKET))

        if packets_to_get > MAX_PACKETS_IN_FRAME:
            raise ValueError("Too many packets to get")

        report = struct.pack(
            "<BBHHB",
            ZERO_REPORT_ID,
            RequestCode.get_frame.value,
            offset,
            buffer_index,
            packets_to_get,
        )

        self.send(report)

        frame_buffer = [0] * pixels_in_frame
        packets_remaining = MAX_PACKETS_IN_FRAME
        packets_received = 0
        while packets_remaining > 0:
            reply = self.receive(ReplyCode.get_frame, STANDARD_TIMEOUT_MS)
            packets_received += 1

            data = struct.unpack(
                "<BHB" + "H" * NUM_OF_PIXELS_IN_PACKET, bytearray(reply)
            )
            pixel_offset = data[1]
            packets_remaining = data[2]
            pixels = data[3:]

            if packets_remaining >= REMAINING_PACKETS_ERROR:
                raise ValueError("Device error when sending packets.")

            if not packets_remaining == (packets_to_get - packets_received):
                raise OSError("Remaining packets error.  Packet dropped?")

            end_offset = pixel_offset + len(pixels)
            frame_buffer[pixel_offset:end_offset] = pixels

        # frame size based on 32 starting, 14 final elements
        data = frame_buffer[32 : pixels_in_frame - 14]
        LOGGER.debug(f"Read {len(data)} pixels")
        return data

    def grab_one(self, exposure_ms=None) -> list:
        if exposure_ms:
            self.parameters.exposure_time_ms = exposure_ms
        self.clear_memory()
        self.set_parameters()
        self.software_trigger()
        # TODO error if this loop makes >2 iterations
        while self.get_status() == Status.in_progress:
            LOGGER.debug("Waiting for capture to finish")
            time.sleep(self.parameters.exposure_time_ms / 1000)
        raw_read = self.get_raw_frame()
        return raw_read

    def _check_flash_parameters(self, data: int | bytearray, offset: int = 0) -> None:
        if isinstance(data, bytearray) or isinstance(data, bytes):
            length = len(data)
        elif isinstance(data, int):
            length = data
        else:
            raise ValueError(f"Unknown flash data type of {type(data)}")

        if length < 0 or offset < 0:
            raise ValueError("Length and offset must be positive")
        if offset > FLASH_MAX_OFFSET:
            raise ValueError(
                f"offset of {offset} greater than maximum of {FLASH_MAX_OFFSET}"
            )
        if offset + length > FLASH_MAX_BYTES:
            raise ValueError(
                f"length + offset of {offset + length} greater than maximum of {FLASH_MAX_BYTES}"
            )

    def read_flash(self, bytes_to_read: int, abs_offset: int = 0) -> bytearray:
        """
        Read the flash memory from the spectrometer

        There are two loops here because there's a limit of
        FLASH_MAX_READ_PACKETS before a subsequent data request can be made.
        """
        self._check_flash_parameters(bytes_to_read, abs_offset)

        payload_size = PACKET_SIZE_BYTES - 4
        packets_to_get = int(math.ceil(bytes_to_read / payload_size))

        buffer = [0] * packets_to_get * payload_size
        offset_increment = 0
        LOGGER.debug(f"Reading {bytes_to_read} bytes from flash")
        while packets_to_get:
            packet_batch = int(min(packets_to_get, FLASH_MAX_READ_PACKETS))

            report = struct.pack(
                "<BBIB",
                ZERO_REPORT_ID,
                RequestCode.read_flash.value,
                abs_offset + offset_increment,
                packet_batch,
            )
            self.send(report)

            packets_remaining = packet_batch
            packets_received = 0
            while packets_remaining > 0:
                reply = self.receive(ReplyCode.read_flash, STANDARD_TIMEOUT_MS)
                packets_received += 1

                data = struct.unpack("<BHB" + "B" * payload_size, bytearray(reply))
                local_offset = data[1]
                packets_remaining = data[2]
                data_frame = data[3:]

                if packets_remaining >= REMAINING_PACKETS_ERROR:
                    raise ValueError("Device error when sending packets.")

                if not packets_remaining == (packet_batch - packets_received):
                    raise OSError("Remaining packets error.  Packet dropped?")

                start_offset = offset_increment + local_offset
                end_offset = start_offset + len(data_frame)
                buffer[start_offset:end_offset] = data_frame

            packets_to_get = max(0, packets_to_get - packet_batch)
            offset_increment += packet_batch * payload_size

        return bytearray(buffer[:bytes_to_read])

    def erase_flash(self) -> None:
        LOGGER.debug("Erasing Flash")
        report = [ZERO_REPORT_ID, RequestCode.erase_flash.value]
        _ = self.send_and_receive(
            report,
            ReplyCode.erase_flash,
            FLASH_ERASE_TIMEOUT_MS,
        )

    def write_flash(self, data_bytes: bytearray, abs_offset: int = 0) -> None:
        self._check_flash_parameters(data_bytes, abs_offset)

        bytes_remaining = len(data_bytes)
        read_offset = 0
        write_offset = abs_offset

        LOGGER.debug(f"Writing {bytes_remaining} bytes to flash")
        while bytes_remaining:
            payload_size = min(bytes_remaining, FLASH_MAX_WRITE_BYTES)
            payload = data_bytes[read_offset : read_offset + payload_size]
            report = struct.pack(
                "<BBIB" + "B" * payload_size,
                ZERO_REPORT_ID,
                RequestCode.write_flash.value,
                write_offset,
                payload_size,
                *payload,
            )
            _ = self.send_and_receive(
                report,
                ReplyCode.write_flash,
                STANDARD_TIMEOUT_MS,
            )

            read_offset += payload_size
            write_offset += payload_size
            bytes_remaining = max(0, bytes_remaining - payload_size)

    def get_calibration(self) -> Calibration:
        LOGGER.debug("Loading calibration")
        # 97089 is the length of the factory calibration
        # Every line had \r\n and every array line had \t\r\n
        # only \n is used here, but the whole length is still read
        BYTES_TO_READ = 97089
        try:
            raw_read = self.read_flash(BYTES_TO_READ, abs_offset=0)
            self.calibration = Calibration().from_bytes(raw_read)
            LOGGER.debug("Calibration loaded")
            return self.calibration
        except Exception as e:
            # Don't fail here to provide chance to load new calibration
            LOGGER.error(f"Unable to load calibration. {e}")

    def apply_irradiance_calibration(self, raw_spectra: np.array) -> np.array:
        """
        First, baseline-correct (dark-frame subtract) the spectra.

        corrected_spectra = (spectra * irr_norm) / (prnu_norm * exp_time * irr_scaler)

        Where:
            spectra = raw data from libspectrometer.so
            irr_norm = irradiance normalization spectra from calibration file
            prnu_norm = photon response non-uniformity normalization. Also captures grating efficiency differences.
            exp_time = exposure time in 10s of uS.
            irr_scaler = correction coefficient from the calibration file
        """
        return np.multiply(raw_spectra, self.calibration.irr_norm) / (
            self.calibration.prnu_norm
            * self.calibration.irr_scaler
            * (self.parameters.exposure_time_ms * 100)  # converted to 10s of uS
        )


def enumerate() -> list[dict]:
    """Returns a list of all connected spectrometers"""
    spectrometers = []
    for device_dict in hid.enumerate():
        if (
            device_dict["vendor_id"] == VENDOR_ID
            and device_dict["product_id"] == PRODUCT_ID
        ):
            spectrometers.append(device_dict)
    return spectrometers


# Example Usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    frame = None
    wave = None
    with LR1.discover() as spectro:
        # spectro.verbose = True
        print(spectro.status)
        wave = spectro.calibration.wavelengths
        frame = spectro.grab_one(200)

    plt.plot(wave, frame)
    plt.show()
