import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5 import QtWidgets, QtCore
from core import DesignerDisplay
import pyqtgraph as pg
import vmbpy as vm
import numpy as np
from skimage import filters
from skimage.measure import regionprops
import matplotlib
from scipy.optimize import curve_fit

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def gaussian_fit(x, A, w, x0):
    return A * np.exp(-2 * ((x - x0) / w) ** 2)


def fit_projection(pixel, intensity, max_evaluations=2000):
    """Fit a 1/e^2 Gaussian radius to a one-dimensional projection."""
    pixel = np.asarray(pixel, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    initial = (
        float(np.max(intensity)),
        max(float(len(pixel)) / 4, 1.0),
        float(np.argmax(intensity)),
    )
    try:
        parameters, _ = curve_fit(
            gaussian_fit,
            pixel,
            intensity,
            p0=initial,
            maxfev=max_evaluations,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None, None

    width = abs(float(parameters[1]))
    return gaussian_fit(pixel, parameters[0], width, parameters[2]), width


@dataclass
class AnalysisResult:
    frame: np.ndarray
    region_analysis_enabled: bool
    row_pixel: np.ndarray
    column_pixel: np.ndarray
    row_sum: np.ndarray
    column_sum: np.ndarray
    row_fit: Optional[np.ndarray]
    column_fit: Optional[np.ndarray]
    row_width: Optional[float]
    column_width: Optional[float]
    centroid: Optional[tuple]
    semimajor: Optional[float]
    semiminor: Optional[float]
    orientation: Optional[float]
    ellipse_x: Optional[np.ndarray]
    ellipse_y: Optional[np.ndarray]


@dataclass(frozen=True)
class AnalysisSettings:
    """A per-frame settings snapshot safe to pass between Qt threads."""

    fit_gaussian: bool = True
    find_region: bool = True
    threshold_mode: str = "Otsu"
    manual_threshold: float = 128.0
    max_fit_evaluations: int = 2000
    flip_x: bool = False
    flip_y: bool = False


@dataclass(frozen=True)
class AnalysisJob:
    frame: np.ndarray
    settings: AnalysisSettings


@dataclass(frozen=True)
class ImageSaveJob:
    frame: np.ndarray
    path: str
    periodic: bool = False


def analyze_frame(frame, settings=None):
    """Perform all CPU-heavy calculations for a camera frame."""
    settings = settings or AnalysisSettings()
    if settings.flip_x:
        frame = np.flip(frame, axis=1)
    if settings.flip_y:
        frame = np.flip(frame, axis=0)
    if settings.flip_x or settings.flip_y:
        frame = np.ascontiguousarray(frame)

    row_sum = np.sum(frame, axis=1)
    column_sum = np.sum(frame, axis=0)
    row_pixel = np.arange(len(row_sum))
    column_pixel = np.arange(len(column_sum))
    if settings.fit_gaussian:
        row_fit, row_width = fit_projection(
            row_pixel, row_sum, settings.max_fit_evaluations
        )
        column_fit, column_width = fit_projection(
            column_pixel, column_sum, settings.max_fit_evaluations
        )
    else:
        row_fit = row_width = None
        column_fit = column_width = None

    centroid = None
    semimajor = None
    semiminor = None
    orientation = None
    ellipse_x = None
    ellipse_y = None

    properties = []
    if settings.find_region:
        if settings.threshold_mode == "Manual":
            threshold = settings.manual_threshold
        elif settings.threshold_mode == "Mean":
            threshold = filters.threshold_mean(frame)
        else:
            threshold = filters.threshold_otsu(frame)
        foreground = (frame > threshold).astype(np.uint8)
        properties = regionprops(foreground, frame)
    if properties:
        prop = properties[0]
        centroid = (
            float(prop.weighted_centroid[1]),
            float(prop.weighted_centroid[0]),
        )
        orientation = -float(prop.orientation + (np.pi / 2))
        semimajor = float(prop.axis_major_length / 2)
        semiminor = float(prop.axis_minor_length / 2)

        angle = np.linspace(0, 2 * np.pi, 100)
        ellipse = np.array(
            [semimajor * np.cos(angle), semiminor * np.sin(angle)]
        )
        rotation = np.array(
            [
                [np.cos(orientation), -np.sin(orientation)],
                [np.sin(orientation), np.cos(orientation)],
            ]
        )
        rotated = rotation @ ellipse
        ellipse_x = centroid[0] + rotated[0]
        ellipse_y = centroid[1] + rotated[1]

    return AnalysisResult(
        frame=frame,
        region_analysis_enabled=settings.find_region,
        row_pixel=row_pixel,
        column_pixel=column_pixel,
        row_sum=row_sum,
        column_sum=column_sum,
        row_fit=row_fit,
        column_fit=column_fit,
        row_width=row_width,
        column_width=column_width,
        centroid=centroid,
        semimajor=semimajor,
        semiminor=semiminor,
        orientation=orientation,
        ellipse_x=ellipse_x,
        ellipse_y=ellipse_y,
    )


class AcquisitionWorker(QtCore.QObject):
    """Acquire frames without blocking Qt's GUI event loop."""

    frame_ready = QtCore.pyqtSignal(object)
    camera_controls_ready = QtCore.pyqtSignal(object)
    camera_settings_applied = QtCore.pyqtSignal(object)
    warning = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, camera_id=None, max_update_hz=30):
        super().__init__()
        self.camera_id = camera_id
        self._rate_lock = threading.Lock()
        self._minimum_emit_interval = 1 / max_update_hz
        self._settings_lock = threading.Lock()
        self._pending_camera_settings = None
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()

    @QtCore.pyqtSlot(float)
    def set_max_update_hz(self, max_update_hz):
        """This slot may be called directly; only mutate lock-protected state."""
        with self._rate_lock:
            self._minimum_emit_interval = 1 / max(float(max_update_hz), 0.1)

    @QtCore.pyqtSlot(object)
    def queue_camera_settings(self, settings):
        """Store only the latest requested values for the acquisition loop."""
        with self._settings_lock:
            self._pending_camera_settings = dict(settings)

    @QtCore.pyqtSlot(bool)
    def set_paused(self, paused):
        """This direct slot only changes thread-safe event state."""
        if paused:
            self._pause_requested.set()
        else:
            self._pause_requested.clear()

    @staticmethod
    def _numeric_feature_info(camera, name):
        try:
            feature = camera.get_feature_by_name(name)
            minimum, maximum = feature.get_range()
            try:
                increment = feature.get_increment()
            except Exception:
                increment = 1
            return {
                "available": True,
                "writable": bool(feature.is_writeable()),
                "minimum": minimum,
                "maximum": maximum,
                "increment": increment,
                "value": feature.get(),
            }
        except Exception:
            return {"available": False, "writable": False}

    def _camera_control_info(self, camera):
        return {
            "exposure": self._numeric_feature_info(camera, "ExposureTime"),
            "binning_x": self._numeric_feature_info(camera, "BinningHorizontal"),
            "binning_y": self._numeric_feature_info(camera, "BinningVertical"),
        }

    @staticmethod
    def _set_numeric_feature(camera, name, requested_value):
        feature = camera.get_feature_by_name(name)
        minimum, maximum = feature.get_range()
        value = min(max(requested_value, minimum), maximum)
        try:
            increment = feature.get_increment()
        except Exception:
            increment = 0
        if increment:
            value = minimum + round((value - minimum) / increment) * increment
        feature.set(value)
        return feature.get()

    def _apply_pending_camera_settings(self, camera):
        with self._settings_lock:
            settings = self._pending_camera_settings
            self._pending_camera_settings = None
        if not settings:
            return

        failures = []
        applied = {}
        if "exposure" in settings:
            try:
                # A numeric exposure request is explicitly a manual setting.
                try:
                    camera.get_feature_by_name("ExposureAuto").set("Off")
                except Exception:
                    pass
                applied["exposure"] = self._set_numeric_feature(
                    camera, "ExposureTime", settings["exposure"]
                )
            except Exception as exc:
                failures.append(f"exposure: {exc}")

        for key, feature_name in (
            ("binning_x", "BinningHorizontal"),
            ("binning_y", "BinningVertical"),
        ):
            if key not in settings:
                continue
            try:
                applied[key] = self._set_numeric_feature(
                    camera, feature_name, settings[key]
                )
            except Exception as exc:
                failures.append(f"{key}: {exc}")

        if applied:
            self.camera_settings_applied.emit(applied)
        if failures:
            self.warning.emit("Camera setting failed — " + "; ".join(failures))

    @QtCore.pyqtSlot()
    def run(self):
        last_emit = 0.0
        try:
            vmb = vm.VmbSystem.get_instance()
            with vmb:
                cameras = vmb.get_all_cameras()
                if not cameras:
                    raise RuntimeError("No Allied Vision camera was detected.")
                camera = (
                    vmb.get_camera_by_id(self.camera_id)
                    if self.camera_id
                    else cameras[0]
                )
                with camera:
                    self.camera_controls_ready.emit(self._camera_control_info(camera))
                    while not self._stop_requested.is_set():
                        while self._pause_requested.is_set():
                            if self._stop_requested.wait(0.05):
                                break
                        if self._stop_requested.is_set():
                            break
                        self._apply_pending_camera_settings(camera)
                        frame = camera.get_frame()
                        now = time.monotonic()
                        with self._rate_lock:
                            minimum_emit_interval = self._minimum_emit_interval
                        if now - last_emit < minimum_emit_interval:
                            continue
                        # Own the data before Vimba recycles its frame buffer.
                        array = frame.as_numpy_ndarray()
                        if array.ndim == 3:
                            array = array[:, :, 0]
                        array = array.copy()
                        self.frame_ready.emit(array)
                        last_emit = now
        except Exception as exc:
            self.error.emit(f"Camera acquisition stopped: {exc}")
        finally:
            self.finished.emit()

    @QtCore.pyqtSlot()
    def request_stop(self):
        self._stop_requested.set()


class CameraDiscoveryWorker(QtCore.QObject):
    """Enumerate cameras outside the GUI thread."""

    cameras_ready = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            with vm.VmbSystem.get_instance() as vmb:
                cameras = []
                for camera in vmb.get_all_cameras():
                    camera_id = camera.get_id()
                    try:
                        label = f"{camera.get_name()} ({camera_id})"
                    except Exception:
                        label = camera_id
                    cameras.append((label, camera_id))
                self.cameras_ready.emit(cameras)
        except Exception as exc:
            self.error.emit(f"Camera discovery failed: {exc}")
        finally:
            self.finished.emit()


class AnalysisWorker(QtCore.QObject):
    """Run image fitting and region analysis on a dedicated thread."""

    result_ready = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot(object)
    def analyze(self, job):
        try:
            self.result_ready.emit(analyze_frame(job.frame, job.settings))
        except Exception as exc:
            self.error.emit(f"Frame analysis failed: {exc}")


class ImageSaveWorker(QtCore.QObject):
    """Serialize NPY writes and keep periodic saving from building a backlog."""

    saved = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._condition = threading.Condition()
        self._manual_jobs = deque()
        self._latest_periodic_job = None
        self._stopping = False

    @QtCore.pyqtSlot(object)
    def enqueue(self, job):
        """Called directly; only enqueue immutable jobs under the condition lock."""
        with self._condition:
            if self._stopping:
                return
            if job.periodic:
                self._latest_periodic_job = job
            else:
                self._manual_jobs.append(job)
            self._condition.notify()

    @QtCore.pyqtSlot()
    def request_stop(self):
        """Drop pending periodic work but finish every explicit manual save."""
        with self._condition:
            self._stopping = True
            self._latest_periodic_job = None
            self._condition.notify()

    @QtCore.pyqtSlot()
    def cancel_periodic(self):
        with self._condition:
            self._latest_periodic_job = None

    @QtCore.pyqtSlot()
    def run(self):
        try:
            while True:
                with self._condition:
                    while (
                        not self._manual_jobs
                        and self._latest_periodic_job is None
                        and not self._stopping
                    ):
                        self._condition.wait()

                    if self._manual_jobs:
                        job = self._manual_jobs.popleft()
                    elif self._stopping:
                        break
                    else:
                        job = self._latest_periodic_job
                        self._latest_periodic_job = None

                try:
                    path = Path(job.path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(path), job.frame, allow_pickle=False)
                    self.saved.emit(job)
                except Exception as exc:
                    self.error.emit(f"Could not save {job.path}: {exc}")
        finally:
            self.finished.emit()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class Viewer(DesignerDisplay, QtWidgets.QWidget):
    filename = "viewer.ui"
    analyze_requested = QtCore.pyqtSignal(object)
    acquisition_rate_changed = QtCore.pyqtSignal(float)
    camera_settings_changed = QtCore.pyqtSignal(object)
    acquisition_pause_changed = QtCore.pyqtSignal(bool)
    acquisition_stop_requested = QtCore.pyqtSignal()
    save_requested = QtCore.pyqtSignal(object)
    periodic_save_cancel_requested = QtCore.pyqtSignal()
    save_stop_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        # self. add_menubar(self.ImageWidget)
        self.left_plot = MplCanvas(self, width=2, height=5)
        self.left_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
        )
        self.top_plot = MplCanvas(self, width=5, height=2)
        self.top_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.left_plot.setFixedWidth(190)
        self.top_plot.setFixedHeight(155)
        self.pixel_um_conversion = 2.2
        self.left_plot.axes.invert_xaxis()
        self.left_plot.axes.invert_yaxis()
        self._build_projection_layout()
        self.curve = pg.PlotCurveItem(pen=pg.mkPen(color="w", width=3))
        self.crosshair = pg.ScatterPlotItem(
            symbol="+", pen=pg.mkPen(color="w", width=3)
        )
        self.ImageView.addItem(self.curve)
        self.ImageView.addItem(self.crosshair)
        self.left_plot_data = self.left_plot.axes.plot([], [])[0]
        self.top_plot_data = self.top_plot.axes.plot([], [])[0]
        self.left_plot_fit = self.left_plot.axes.plot([], [])[0]
        self.top_plot_fit = self.top_plot.axes.plot([], [])[0]
        self._analysis_busy = False
        self._pending_frame = None
        self._stopping = False
        self._restart_acquisition = False
        self.acquisition_thread = None
        self.acquisition_worker = None
        self.discovery_thread = None
        self.discovery_worker = None
        self._auto_start_after_discovery = False
        self._current_frame = None
        self._periodic_save_deadline = None
        self._save_sequence = 0
        self.RefGroupBox.setVisible(False)
        self._build_runtime_controls()
        self._build_save_controls()
        self._start_workers()

    def _build_projection_layout(self):
        """Align each projection canvas with the matching image dimension."""
        # Use the full canvas extent along the shared dimension. This makes the
        # top X axis and side Y axis physically span the same width/height as
        # the image widget, in addition to having matching numeric limits.
        self.top_plot.figure.subplots_adjust(
            left=0.0, right=1.0, bottom=0.25, top=0.95
        )
        self.left_plot.figure.subplots_adjust(
            left=0.28, right=0.95, bottom=0.0, top=1.0
        )

        # The Designer layout placed the side plot beside a container holding
        # the top plot, image, and status bar, so its height could never match
        # the image. Rebuild that portion as one shared grid.
        first_item = self.horizontalLayout.takeAt(0)
        old_left_layout = first_item.layout() if first_item is not None else None
        if old_left_layout is not None:
            old_left_layout.setParent(None)
            old_left_layout.deleteLater()

        while self.verticalLayout_3.count():
            item = self.verticalLayout_3.takeAt(0)
            child_layout = item.layout()
            if child_layout is not None:
                child_layout.setParent(None)
                child_layout.deleteLater()

        self.ImageWidget.hide()
        projection_widget = QtWidgets.QWidget(self.frame_4)
        projection_grid = QtWidgets.QGridLayout(projection_widget)
        projection_grid.setContentsMargins(0, 0, 0, 0)
        projection_grid.setHorizontalSpacing(6)
        projection_grid.setVerticalSpacing(6)
        projection_grid.addWidget(self.top_plot, 0, 1)
        projection_grid.addWidget(self.left_plot, 1, 0)
        projection_grid.addWidget(self.ImageView, 1, 1)
        projection_grid.addWidget(self.frame_3, 2, 1)
        projection_grid.setColumnStretch(0, 0)
        projection_grid.setColumnStretch(1, 1)
        projection_grid.setRowStretch(0, 0)
        projection_grid.setRowStretch(1, 1)
        projection_grid.setRowStretch(2, 0)
        self.verticalLayout_3.addWidget(projection_widget)

    def _build_runtime_controls(self):
        controls = QtWidgets.QGroupBox("Runtime controls", self.ImageGroupBox)
        layout = QtWidgets.QGridLayout(controls)

        self.CameraComboBox = QtWidgets.QComboBox(controls)
        self.CameraComboBox.addItem("First available camera", None)
        self.CameraComboBox.setToolTip("Changing camera restarts acquisition safely.")
        self.RefreshCamerasButton = QtWidgets.QPushButton("Refresh", controls)
        self.StartButton = QtWidgets.QPushButton("Start", controls)
        self.StopButton = QtWidgets.QPushButton("Stop", controls)
        self.StopButton.setEnabled(False)
        self.PauseButton = QtWidgets.QPushButton("Pause", controls)
        self.PauseButton.setCheckable(True)
        self.PauseButton.setEnabled(False)
        self.CameraStatusLabel = QtWidgets.QLabel("Discovering cameras…", controls)
        self.CameraStatusLabel.setMinimumWidth(150)

        self.UpdateRateSpinBox = QtWidgets.QDoubleSpinBox(controls)
        self.UpdateRateSpinBox.setRange(0.1, 240.0)
        self.UpdateRateSpinBox.setDecimals(1)
        self.UpdateRateSpinBox.setValue(30.0)
        self.UpdateRateSpinBox.setSuffix(" Hz")
        self.UpdateRateSpinBox.setToolTip(
            "Maximum rate sent to analysis and display; camera acquisition may be faster."
        )

        self.PixelSizeSpinBox = QtWidgets.QDoubleSpinBox(controls)
        self.PixelSizeSpinBox.setRange(0.001, 1000.0)
        self.PixelSizeSpinBox.setDecimals(3)
        self.PixelSizeSpinBox.setValue(self.pixel_um_conversion)
        self.PixelSizeSpinBox.setSuffix(" µm/px")
        self.PixelSizeSpinBox.setToolTip("Calibration used for all millimetre readouts.")

        self.ThresholdComboBox = QtWidgets.QComboBox(controls)
        self.ThresholdComboBox.addItems(["Otsu", "Mean", "Manual"])
        self.ManualThresholdSpinBox = QtWidgets.QDoubleSpinBox(controls)
        self.ManualThresholdSpinBox.setRange(0.0, 65535.0)
        self.ManualThresholdSpinBox.setValue(128.0)
        self.ManualThresholdSpinBox.setEnabled(False)
        self.ManualThresholdSpinBox.setToolTip(
            "Foreground pixels must be greater than this value."
        )

        self.FitEvaluationsSpinBox = QtWidgets.QSpinBox(controls)
        self.FitEvaluationsSpinBox.setRange(100, 100000)
        self.FitEvaluationsSpinBox.setSingleStep(500)
        self.FitEvaluationsSpinBox.setValue(2000)
        self.FitEvaluationsSpinBox.setToolTip(
            "Maximum solver evaluations per projection; higher values can improve "
            "difficult fits but take longer."
        )

        self.GaussianFitCheckBox = QtWidgets.QCheckBox("Gaussian fits", controls)
        self.GaussianFitCheckBox.setChecked(True)
        self.RegionAnalysisCheckBox = QtWidgets.QCheckBox("Centroid / ellipse", controls)
        self.RegionAnalysisCheckBox.setChecked(True)
        self.FlipXCheckBox = QtWidgets.QCheckBox("Flip X", controls)
        self.FlipYCheckBox = QtWidgets.QCheckBox("Flip Y", controls)

        self.ExposureSpinBox = QtWidgets.QDoubleSpinBox(controls)
        self.ExposureSpinBox.setRange(1.0, 10000000.0)
        self.ExposureSpinBox.setDecimals(2)
        self.ExposureSpinBox.setSuffix(" µs")
        self.ExposureSpinBox.setEnabled(False)
        self.ExposureSpinBox.setToolTip(
            "Manual camera exposure time. Its supported range is read from the camera."
        )
        self.BinningXSpinBox = QtWidgets.QSpinBox(controls)
        self.BinningYSpinBox = QtWidgets.QSpinBox(controls)
        for spin_box in (self.BinningXSpinBox, self.BinningYSpinBox):
            spin_box.setRange(1, 16)
            spin_box.setEnabled(False)
            spin_box.setToolTip(
                "Sensor binning. Its supported range is read from the camera."
            )

        layout.addWidget(QtWidgets.QLabel("Camera:"), 0, 0)
        layout.addWidget(self.CameraComboBox, 0, 1, 1, 3)
        layout.addWidget(self.RefreshCamerasButton, 0, 4)
        layout.addWidget(self.StartButton, 0, 5)
        layout.addWidget(self.StopButton, 0, 6)
        layout.addWidget(self.PauseButton, 0, 7)
        layout.addWidget(self.CameraStatusLabel, 0, 8)
        layout.addWidget(QtWidgets.QLabel("Display rate:"), 1, 0)
        layout.addWidget(self.UpdateRateSpinBox, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Pixel size:"), 1, 2)
        layout.addWidget(self.PixelSizeSpinBox, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Threshold:"), 1, 4)
        layout.addWidget(self.ThresholdComboBox, 1, 5)
        layout.addWidget(self.ManualThresholdSpinBox, 1, 6)
        layout.addWidget(self.GaussianFitCheckBox, 2, 0, 1, 2)
        layout.addWidget(self.RegionAnalysisCheckBox, 2, 2, 1, 2)
        layout.addWidget(self.FlipXCheckBox, 2, 4)
        layout.addWidget(self.FlipYCheckBox, 2, 5)
        layout.addWidget(QtWidgets.QLabel("Fit evaluations:"), 2, 6)
        layout.addWidget(self.FitEvaluationsSpinBox, 2, 7)
        layout.addWidget(QtWidgets.QLabel("Exposure:"), 3, 0)
        layout.addWidget(self.ExposureSpinBox, 3, 1)
        layout.addWidget(QtWidgets.QLabel("Horizontal binning:"), 3, 2)
        layout.addWidget(self.BinningXSpinBox, 3, 3)
        layout.addWidget(QtWidgets.QLabel("Vertical binning:"), 3, 4)
        layout.addWidget(self.BinningYSpinBox, 3, 5)
        layout.setColumnStretch(8, 1)
        self.verticalLayout_8.insertWidget(0, controls)

        self.RefreshCamerasButton.clicked.connect(self._discover_cameras)
        self.StartButton.clicked.connect(self._start_acquisition)
        self.StopButton.clicked.connect(self._stop_acquisition)
        self.PauseButton.toggled.connect(self._toggle_acquisition_pause)
        self.CameraComboBox.currentIndexChanged.connect(self._camera_changed)
        self.UpdateRateSpinBox.valueChanged.connect(self._update_rate_changed)
        self.PixelSizeSpinBox.valueChanged.connect(self._pixel_size_changed)
        self.ThresholdComboBox.currentTextChanged.connect(
            lambda text: self.ManualThresholdSpinBox.setEnabled(text == "Manual")
        )
        self.ExposureSpinBox.valueChanged.connect(self._camera_settings_changed)
        self.BinningXSpinBox.valueChanged.connect(self._camera_settings_changed)
        self.BinningYSpinBox.valueChanged.connect(self._camera_settings_changed)
        self._update_rate_changed(self.UpdateRateSpinBox.value())

    def _build_save_controls(self):
        controls = QtWidgets.QGroupBox("NPY image saving", self.ImageGroupBox)
        layout = QtWidgets.QGridLayout(controls)

        self.PeriodicSaveCheckBox = QtWidgets.QCheckBox("Save periodically", controls)
        self.SaveIntervalSpinBox = QtWidgets.QDoubleSpinBox(controls)
        self.SaveIntervalSpinBox.setRange(0.1, 86400.0)
        self.SaveIntervalSpinBox.setDecimals(1)
        self.SaveIntervalSpinBox.setValue(10.0)
        self.SaveIntervalSpinBox.setSuffix(" s")
        self.SaveIntervalSpinBox.setToolTip("Elapsed time between saved image arrays.")
        self.SaveDirectoryLineEdit = QtWidgets.QLineEdit(controls)
        self.SaveDirectoryLineEdit.setText(str(Path.cwd() / "captures"))
        self.SaveDirectoryLineEdit.setToolTip(
            "Periodic images are written here as timestamped .npy files."
        )
        self.BrowseSaveDirectoryButton = QtWidgets.QPushButton("Browse…", controls)
        self.SaveCurrentButton = QtWidgets.QPushButton("Save current image…", controls)
        self.SaveCurrentButton.setEnabled(False)
        self.SaveStatusLabel = QtWidgets.QLabel("No image saved yet", controls)

        layout.addWidget(self.PeriodicSaveCheckBox, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Interval:"), 0, 1)
        layout.addWidget(self.SaveIntervalSpinBox, 0, 2)
        layout.addWidget(QtWidgets.QLabel("Folder:"), 0, 3)
        layout.addWidget(self.SaveDirectoryLineEdit, 0, 4)
        layout.addWidget(self.BrowseSaveDirectoryButton, 0, 5)
        layout.addWidget(self.SaveCurrentButton, 0, 6)
        layout.addWidget(self.SaveStatusLabel, 1, 0, 1, 7)
        layout.setColumnStretch(4, 1)
        self.verticalLayout_8.insertWidget(1, controls)

        self.PeriodicSaveCheckBox.toggled.connect(self._periodic_save_changed)
        self.SaveIntervalSpinBox.valueChanged.connect(self._periodic_save_changed)
        self.BrowseSaveDirectoryButton.clicked.connect(self._browse_save_directory)
        self.SaveCurrentButton.clicked.connect(self._save_current_image)

    def _set_camera_controls_enabled(self, enabled):
        for spin_box in (
            self.ExposureSpinBox,
            self.BinningXSpinBox,
            self.BinningYSpinBox,
        ):
            spin_box.setEnabled(enabled and spin_box.property("cameraWritable") is True)

    @QtCore.pyqtSlot(object)
    def _configure_camera_controls(self, controls):
        if self.acquisition_thread is None or not self.acquisition_thread.isRunning():
            return
        specifications = (
            (self.ExposureSpinBox, controls["exposure"], False),
            (self.BinningXSpinBox, controls["binning_x"], True),
            (self.BinningYSpinBox, controls["binning_y"], True),
        )
        for spin_box, feature, integer_value in specifications:
            writable = feature.get("available", False) and feature.get(
                "writable", False
            )
            spin_box.setProperty("cameraWritable", writable)
            spin_box.blockSignals(True)
            if feature.get("available", False):
                minimum = feature["minimum"]
                maximum = feature["maximum"]
                increment = feature.get("increment") or 1
                if integer_value:
                    spin_box.setRange(int(minimum), int(maximum))
                    spin_box.setSingleStep(max(1, int(increment)))
                    spin_box.setValue(int(feature["value"]))
                else:
                    spin_box.setRange(float(minimum), float(maximum))
                    spin_box.setSingleStep(max(float(increment), 0.001))
                    spin_box.setValue(float(feature["value"]))
                spin_box.setToolTip(
                    f"Camera range: {minimum:g} to {maximum:g}; "
                    f"increment: {increment:g}."
                )
            else:
                spin_box.setToolTip("This feature is not supported by the selected camera.")
            spin_box.blockSignals(False)
            spin_box.setEnabled(writable)

    @QtCore.pyqtSlot()
    def _camera_settings_changed(self):
        settings = {}
        if self.ExposureSpinBox.isEnabled():
            settings["exposure"] = self.ExposureSpinBox.value()
        if self.BinningXSpinBox.isEnabled():
            settings["binning_x"] = self.BinningXSpinBox.value()
        if self.BinningYSpinBox.isEnabled():
            settings["binning_y"] = self.BinningYSpinBox.value()
        if settings:
            self.camera_settings_changed.emit(settings)

    @QtCore.pyqtSlot(object)
    def _update_applied_camera_values(self, values):
        widgets = {
            "exposure": self.ExposureSpinBox,
            "binning_x": self.BinningXSpinBox,
            "binning_y": self.BinningYSpinBox,
        }
        for key, value in values.items():
            spin_box = widgets[key]
            spin_box.blockSignals(True)
            spin_box.setValue(value)
            spin_box.blockSignals(False)
        self.CameraStatusLabel.setText(
            "Paused" if self.PauseButton.isChecked() else "Running"
        )

    @QtCore.pyqtSlot(str)
    def _show_camera_warning(self, message):
        self.CameraStatusLabel.setText("Setting rejected")
        self.MiscLabel.setText(message)

    def _analysis_settings(self):
        return AnalysisSettings(
            fit_gaussian=self.GaussianFitCheckBox.isChecked(),
            find_region=self.RegionAnalysisCheckBox.isChecked(),
            threshold_mode=self.ThresholdComboBox.currentText(),
            manual_threshold=self.ManualThresholdSpinBox.value(),
            max_fit_evaluations=self.FitEvaluationsSpinBox.value(),
            flip_x=self.FlipXCheckBox.isChecked(),
            flip_y=self.FlipYCheckBox.isChecked(),
        )

    @QtCore.pyqtSlot(float)
    def _pixel_size_changed(self, value):
        self.pixel_um_conversion = value

    @QtCore.pyqtSlot(float)
    def _update_rate_changed(self, value):
        self.ImageView.setProperty("maxRedrawRate", max(1, round(value)))
        self.acquisition_rate_changed.emit(value)

    @QtCore.pyqtSlot(bool)
    def _toggle_acquisition_pause(self, paused):
        running = (
            self.acquisition_thread is not None
            and self.acquisition_thread.isRunning()
        )
        if not running:
            self.PauseButton.blockSignals(True)
            self.PauseButton.setChecked(False)
            self.PauseButton.blockSignals(False)
            self.PauseButton.setText("Pause")
            return
        self.PauseButton.setText("Continue" if paused else "Pause")
        self.CameraStatusLabel.setText("Paused" if paused else "Running")
        self.acquisition_pause_changed.emit(paused)

    def _periodic_save_changed(self, _value=None):
        if self.PeriodicSaveCheckBox.isChecked():
            self._periodic_save_deadline = (
                time.monotonic() + self.SaveIntervalSpinBox.value()
            )
            self.SaveStatusLabel.setText("Periodic saving enabled")
        else:
            self._periodic_save_deadline = None
            self.periodic_save_cancel_requested.emit()

    @QtCore.pyqtSlot()
    def _browse_save_directory(self):
        current = Path(self.SaveDirectoryLineEdit.text()).expanduser()
        start_directory = current if current.is_dir() else Path.cwd()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select periodic image folder",
            str(start_directory),
        )
        if directory:
            self.SaveDirectoryLineEdit.setText(directory)

    def _timestamped_save_path(self, directory):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        milliseconds = now.microsecond // 1000
        self._save_sequence += 1
        filename = (
            f"vimba_{timestamp}_{milliseconds:03d}_{self._save_sequence:06d}.npy"
        )
        return Path(directory) / filename

    @QtCore.pyqtSlot()
    def _save_current_image(self):
        if self._current_frame is None:
            self.SaveStatusLabel.setText("No image is available to save")
            return
        directory = Path(self.SaveDirectoryLineEdit.text()).expanduser()
        if not directory.is_dir():
            directory = Path.cwd()
        suggested_path = self._timestamped_save_path(directory)
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save current image array",
            str(suggested_path),
            "NumPy array (*.npy)",
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".npy":
            path = Path(str(path) + ".npy")
        job = ImageSaveJob(self._current_frame.copy(), str(path), periodic=False)
        self.save_requested.emit(job)
        self.SaveStatusLabel.setText(f"Queued manual save: {path.name}")

    def _maybe_save_periodically(self, frame):
        if not self.PeriodicSaveCheckBox.isChecked():
            return
        now = time.monotonic()
        if self._periodic_save_deadline is None:
            self._periodic_save_deadline = now + self.SaveIntervalSpinBox.value()
            return
        if now < self._periodic_save_deadline:
            return

        interval = self.SaveIntervalSpinBox.value()
        while self._periodic_save_deadline <= now:
            self._periodic_save_deadline += interval
        directory_text = self.SaveDirectoryLineEdit.text().strip()
        if not directory_text:
            self.PeriodicSaveCheckBox.setChecked(False)
            self.SaveStatusLabel.setText("Periodic saving stopped: choose a folder")
            return
        path = self._timestamped_save_path(Path(directory_text).expanduser())
        self.save_requested.emit(ImageSaveJob(frame.copy(), str(path), periodic=True))
        self.SaveStatusLabel.setText(f"Queued periodic save: {path.name}")

    @QtCore.pyqtSlot(object)
    def _image_saved(self, job):
        save_kind = "Periodic" if job.periodic else "Manual"
        self.SaveStatusLabel.setText(f"{save_kind} save complete: {job.path}")

    @QtCore.pyqtSlot(str)
    def _image_save_failed(self, message):
        self.SaveStatusLabel.setText(message)

    def _start_workers(self):
        self.save_thread = QtCore.QThread(self)
        self.save_worker = ImageSaveWorker()
        self.save_worker.moveToThread(self.save_thread)
        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.saved.connect(self._image_saved)
        self.save_worker.error.connect(self._image_save_failed)
        self.save_worker.finished.connect(
            self.save_thread.quit,
            type=QtCore.Qt.DirectConnection,
        )
        self.save_worker.finished.connect(self.save_worker.deleteLater)
        self.save_thread.finished.connect(self.save_thread.deleteLater)
        self.save_requested.connect(
            self.save_worker.enqueue,
            type=QtCore.Qt.DirectConnection,
        )
        self.save_stop_requested.connect(
            self.save_worker.request_stop,
            type=QtCore.Qt.DirectConnection,
        )
        self.periodic_save_cancel_requested.connect(
            self.save_worker.cancel_periodic,
            type=QtCore.Qt.DirectConnection,
        )
        self.save_thread.start()

        self.analysis_thread = QtCore.QThread(self)
        self.analysis_worker = AnalysisWorker()
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analyze_requested.connect(self.analysis_worker.analyze)
        self.analysis_worker.result_ready.connect(self._render_result)
        self.analysis_worker.error.connect(self._analysis_failed)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.start()

        self._discover_cameras(auto_start=True)

    def _discover_cameras(self, auto_start=False):
        if self._stopping or (
            self.discovery_thread is not None and self.discovery_thread.isRunning()
        ):
            return
        if self.acquisition_thread is not None and self.acquisition_thread.isRunning():
            self.CameraStatusLabel.setText("Stop acquisition before refreshing")
            return

        self.RefreshCamerasButton.setEnabled(False)
        self.StartButton.setEnabled(False)
        self.CameraStatusLabel.setText("Discovering cameras…")
        self.discovery_thread = QtCore.QThread(self)
        self.discovery_worker = CameraDiscoveryWorker()
        self.discovery_worker.moveToThread(self.discovery_thread)
        self.discovery_thread.started.connect(self.discovery_worker.run)
        self.discovery_worker.cameras_ready.connect(self._cameras_discovered)
        self.discovery_worker.error.connect(self._show_error)
        self.discovery_worker.finished.connect(self.discovery_thread.quit)
        self.discovery_worker.finished.connect(self.discovery_worker.deleteLater)
        self.discovery_thread.finished.connect(self._discovery_finished)
        self.discovery_thread.finished.connect(self.discovery_thread.deleteLater)
        self._auto_start_after_discovery = auto_start
        self.discovery_thread.start()

    @QtCore.pyqtSlot(object)
    def _cameras_discovered(self, cameras):
        selected_id = self.CameraComboBox.currentData()
        self.CameraComboBox.blockSignals(True)
        self.CameraComboBox.clear()
        self.CameraComboBox.addItem("First available camera", None)
        for label, camera_id in cameras:
            self.CameraComboBox.addItem(label, camera_id)
        selected_index = self.CameraComboBox.findData(selected_id)
        self.CameraComboBox.setCurrentIndex(max(selected_index, 0))
        self.CameraComboBox.blockSignals(False)
        self.CameraStatusLabel.setText(
            f"{len(cameras)} camera(s) found" if cameras else "No cameras found"
        )

    @QtCore.pyqtSlot()
    def _discovery_finished(self):
        self.RefreshCamerasButton.setEnabled(True)
        self.StartButton.setEnabled(True)
        self.discovery_thread = None
        self.discovery_worker = None
        if self._auto_start_after_discovery and not self._stopping:
            self._auto_start_after_discovery = False
            self._start_acquisition()

    @QtCore.pyqtSlot()
    def _start_acquisition(self):
        if self._stopping:
            return
        if self.discovery_thread is not None and self.discovery_thread.isRunning():
            return
        if self.acquisition_thread is not None and self.acquisition_thread.isRunning():
            return

        self.acquisition_thread = QtCore.QThread(self)
        self.acquisition_worker = AcquisitionWorker(
            camera_id=self.CameraComboBox.currentData(),
            max_update_hz=self.UpdateRateSpinBox.value(),
        )
        self.acquisition_worker.moveToThread(self.acquisition_thread)
        self.acquisition_thread.started.connect(self.acquisition_worker.run)
        self.acquisition_worker.frame_ready.connect(self._queue_frame)
        self.acquisition_worker.camera_controls_ready.connect(
            self._configure_camera_controls
        )
        self.acquisition_worker.camera_settings_applied.connect(
            self._update_applied_camera_values
        )
        self.acquisition_worker.warning.connect(self._show_camera_warning)
        self.acquisition_worker.error.connect(self._show_error)
        self.acquisition_worker.finished.connect(self.acquisition_thread.quit)
        self.acquisition_worker.finished.connect(self.acquisition_worker.deleteLater)
        self.acquisition_thread.finished.connect(self._acquisition_finished)
        self.acquisition_thread.finished.connect(self.acquisition_thread.deleteLater)
        self.acquisition_rate_changed.connect(
            self.acquisition_worker.set_max_update_hz,
            type=QtCore.Qt.DirectConnection,
        )
        self.acquisition_stop_requested.connect(
            self.acquisition_worker.request_stop,
            type=QtCore.Qt.DirectConnection,
        )
        self.acquisition_pause_changed.connect(
            self.acquisition_worker.set_paused,
            type=QtCore.Qt.DirectConnection,
        )
        self.camera_settings_changed.connect(
            self.acquisition_worker.queue_camera_settings,
            type=QtCore.Qt.DirectConnection,
        )
        self._set_camera_controls_enabled(False)
        self.PauseButton.blockSignals(True)
        self.PauseButton.setChecked(False)
        self.PauseButton.blockSignals(False)
        self.PauseButton.setText("Pause")
        self.PauseButton.setEnabled(True)
        self.StartButton.setEnabled(False)
        self.StopButton.setEnabled(True)
        self.RefreshCamerasButton.setEnabled(False)
        self.CameraStatusLabel.setText("Running")
        self.acquisition_thread.start()

    @QtCore.pyqtSlot()
    def _stop_acquisition(self):
        self._restart_acquisition = False
        if self.acquisition_thread is not None and self.acquisition_thread.isRunning():
            self.CameraStatusLabel.setText("Stopping…")
            self.StopButton.setEnabled(False)
            self.PauseButton.setEnabled(False)
            self.acquisition_stop_requested.emit()
        else:
            self.StartButton.setEnabled(True)
            self.StopButton.setEnabled(False)
            self.PauseButton.setEnabled(False)
            self.CameraStatusLabel.setText("Stopped")

    @QtCore.pyqtSlot()
    def _camera_changed(self):
        if self.acquisition_thread is not None and self.acquisition_thread.isRunning():
            self._restart_acquisition = True
            self.CameraStatusLabel.setText("Switching camera…")
            self.PauseButton.setEnabled(False)
            self.acquisition_stop_requested.emit()

    @QtCore.pyqtSlot()
    def _acquisition_finished(self):
        try:
            self.acquisition_rate_changed.disconnect(self.acquisition_worker.set_max_update_hz)
            self.acquisition_stop_requested.disconnect(self.acquisition_worker.request_stop)
            self.acquisition_pause_changed.disconnect(
                self.acquisition_worker.set_paused
            )
            self.camera_settings_changed.disconnect(
                self.acquisition_worker.queue_camera_settings
            )
        except (TypeError, RuntimeError):
            pass
        self.acquisition_thread = None
        self.acquisition_worker = None
        self.StartButton.setEnabled(True)
        self.StopButton.setEnabled(False)
        self.PauseButton.blockSignals(True)
        self.PauseButton.setChecked(False)
        self.PauseButton.blockSignals(False)
        self.PauseButton.setText("Pause")
        self.PauseButton.setEnabled(False)
        self.RefreshCamerasButton.setEnabled(True)
        self._set_camera_controls_enabled(False)
        if self._restart_acquisition and not self._stopping:
            self._restart_acquisition = False
            self._start_acquisition()
        elif not self._stopping:
            self.CameraStatusLabel.setText("Stopped")

    @QtCore.pyqtSlot(object)
    def _queue_frame(self, frame):
        if self._stopping:
            return
        if self._analysis_busy:
            self._pending_frame = AnalysisJob(frame, self._analysis_settings())
            return
        self._analysis_busy = True
        self.analyze_requested.emit(AnalysisJob(frame, self._analysis_settings()))

    @QtCore.pyqtSlot(object)
    def _render_result(self, result):
        if self._stopping:
            return

        self._current_frame = result.frame
        self.SaveCurrentButton.setEnabled(True)
        self._maybe_save_periodically(result.frame)
        self.ImageView.setImage(result.frame)
        self.left_plot_data.set_data(result.row_sum, result.row_pixel)
        self.top_plot_data.set_data(result.column_pixel, result.column_sum)
        self.left_plot_fit.set_data(
            result.row_fit if result.row_fit is not None else [],
            result.row_pixel if result.row_fit is not None else [],
        )
        self.top_plot_fit.set_data(
            result.column_pixel if result.column_fit is not None else [],
            result.column_fit if result.column_fit is not None else [],
        )

        row_max = max(float(np.max(result.row_sum)), 1.0)
        column_max = max(float(np.max(result.column_sum)), 1.0)
        self.left_plot.axes.set_xlim(row_max * 1.05, 0)
        # Pixel coordinates refer to centers. Half-pixel bounds give each
        # projection axis the same full pixel extent as the image.
        self.left_plot.axes.set_ylim(
            len(result.row_pixel) - 0.5,
            -0.5,
        )
        self.top_plot.axes.set_xlim(
            -0.5,
            len(result.column_pixel) - 0.5,
        )
        self.top_plot.axes.set_ylim(0, column_max * 1.05)
        self.left_plot.draw_idle()
        self.top_plot.draw_idle()

        self._set_width_labels(result)
        self._set_region_labels(result)
        self.MiscLabel.setText("")
        self._analysis_busy = False
        self._dispatch_pending_frame()

    def _set_width_labels(self, result):
        if result.row_width is not None:
            self.wyPixelLabel.setText(f"{result.row_width:.2f}")
            self.wymmLabel.setText(
                f"{result.row_width * self.pixel_um_conversion / 1000:.2f}"
            )
        else:
            self.wyPixelLabel.setText("--")
            self.wymmLabel.setText("--")
        if result.column_width is not None:
            self.wxPixelLabel.setText(f"{result.column_width:.2f}")
            self.wxmmLabel.setText(
                f"{result.column_width * self.pixel_um_conversion / 1000:.2f}"
            )
        else:
            self.wxPixelLabel.setText("--")
            self.wxmmLabel.setText("--")

    def _set_region_labels(self, result):
        if result.centroid is None:
            self.curve.setData([], [])
            self.crosshair.setData(pos=[])
            self.CentroidLabel.setText(
                "No foreground detected"
                if result.region_analysis_enabled
                else "Analysis disabled"
            )
            self.SemimajorLabel.setText("--")
            self.SemiminorLabel.setText("--")
            self.OrientationLabel.setText("--")
            return

        com_x, com_y = result.centroid
        self.curve.setData(result.ellipse_x, result.ellipse_y)
        self.crosshair.setData(pos=[(com_x, com_y)])
        self.CentroidLabel.setText(
            f"Pixel ({com_x:.2f}, {com_y:.2f}); mm "
            f"({com_x * self.pixel_um_conversion / 1000:.2f}, "
            f"{com_y * self.pixel_um_conversion / 1000:.2f})"
        )
        self.SemimajorLabel.setText(
            f"Pixel: {result.semimajor:.2f}; mm: "
            f"{result.semimajor * self.pixel_um_conversion / 1000:.2f}"
        )
        self.SemiminorLabel.setText(
            f"Pixel: {result.semiminor:.2f}; mm: "
            f"{result.semiminor * self.pixel_um_conversion / 1000:.2f}"
        )
        self.OrientationLabel.setText(f"{result.orientation:.2f}")

    @QtCore.pyqtSlot(str)
    def _analysis_failed(self, message):
        self._show_error(message)
        self._analysis_busy = False
        self._dispatch_pending_frame()

    def _dispatch_pending_frame(self):
        if self._pending_frame is None or self._stopping:
            return
        job = self._pending_frame
        self._pending_frame = None
        self._analysis_busy = True
        self.analyze_requested.emit(job)

    @QtCore.pyqtSlot(str)
    def _show_error(self, message):
        self.MiscLabel.setText(message)

    def closeEvent(self, event):
        self._stopping = True
        self._pending_frame = None
        self.PeriodicSaveCheckBox.setChecked(False)
        self.save_stop_requested.emit()
        if self.acquisition_thread is not None and self.acquisition_thread.isRunning():
            self.acquisition_stop_requested.emit()
        self.analysis_thread.quit()
        if self.acquisition_thread is not None:
            self.acquisition_thread.wait(5000)
        if self.discovery_thread is not None:
            self.discovery_thread.quit()
            self.discovery_thread.wait(5000)
        self.analysis_thread.wait(5000)
        self.save_thread.wait()
        super().closeEvent(event)

    # def add_menubar(self, widget: QtWidgets.QWidget):
    #     """
    #     Method to generate menubar for ref viewers and append to screen

    #     Parameters
    #     ----------
    #     widget : QtWidgets.QWidget
    #         blank QWidget in the screen to add the menubar to
    #     """
    #     widget.menuBar = QtWidgets.QMenuBar(widget)
    #     ref_image_menu = widget.menuBar.addMenu("Image")
    #     self.ref_settings_action = QtWidgets.QAction("Settings", widget)
    #     # self.ref_settings_action.triggered.connect(lambda: pass)
    #     ref_image_menu.addAction(self.ref_settings_action)
    #     self.save_ref_action = QtWidgets.QAction("Save", widget)
    #     # self.save_ref_action.triggered.connect(self.save_image)
    #     ref_image_menu.addAction(self.save_ref_action)
    #     self.load_ref_action = QtWidgets.QAction("Load", widget)
    #     # self.load_ref_action.triggered.connect(self.upload_reference)
    #     ref_image_menu.addAction(self.load_ref_action)
    #     ref_overlay_menu = widget.menuBar.addMenu("Overlay")
    #     self.refoverlay_showhide_action = QtWidgets.QAction("Show", widget)
    #     self.refoverlay_showhide_action.setCheckable(True)
    #     self.refoverlay_showhide_action.setChecked(True)
    #     # self.refoverlay_showhide_action.triggered.connect(self.toggle_overlay)
    #     ref_overlay_menu.addAction(self.refoverlay_showhide_action)
    #     self.refoverlay_live_showhide_action = QtWidgets.QAction(
    #         "Show on live cam", widget
    #     )
    #     self.refoverlay_live_showhide_action.setCheckable(True)
    #     self.refoverlay_live_showhide_action.setChecked(False)
    #     # self.refoverlay_live_showhide_action.triggered.connect(
    #     #     self.toggle_live_overlay
    #     # )
    #     ref_overlay_menu.addAction(self.refoverlay_live_showhide_action)
    #     ref_alignment_menu = widget.menuBar.addMenu("Alignment")
    #     self.ref_computedet_action = QtWidgets.QAction("Compute Determination", widget)
    #     self.ref_computedet_action.setCheckable(True)
    #     # self.ref_computedet_action.triggered.connect(
    #     #     self.toggle_determination_computation
    #     # )
    #     ref_alignment_menu.addAction(self.ref_computedet_action)
    #     widget.menuBar.setMinimumWidth(widget.width())
    #     widget.menuBar.show()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication([])
    screen = Viewer()
    # screen.resize(2200, 3400)
    screen.showMaximized()

    qapp.exec_()
