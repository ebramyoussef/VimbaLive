import threading
import time
from dataclasses import dataclass
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


def fit_projection(pixel, intensity):
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
            maxfev=2000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None, None

    width = abs(float(parameters[1]))
    return gaussian_fit(pixel, parameters[0], width, parameters[2]), width


@dataclass
class AnalysisResult:
    frame: np.ndarray
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


def analyze_frame(frame):
    """Perform all CPU-heavy calculations for a camera frame."""
    row_sum = np.sum(frame, axis=1)
    column_sum = np.sum(frame, axis=0)
    row_pixel = np.arange(len(row_sum))
    column_pixel = np.arange(len(column_sum))
    row_fit, row_width = fit_projection(row_pixel, row_sum)
    column_fit, column_width = fit_projection(column_pixel, column_sum)

    centroid = None
    semimajor = None
    semiminor = None
    orientation = None
    ellipse_x = None
    ellipse_y = None

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
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, max_update_hz=30):
        super().__init__()
        self.minimum_emit_interval = 1 / max_update_hz
        self._stop_requested = threading.Event()

    @QtCore.pyqtSlot()
    def run(self):
        last_emit = 0.0
        try:
            vmb = vm.VmbSystem.get_instance()
            with vmb:
                cameras = vmb.get_all_cameras()
                if not cameras:
                    raise RuntimeError("No Allied Vision camera was detected.")
                with cameras[0] as camera:
                    while not self._stop_requested.is_set():
                        frame = camera.get_frame()
                        now = time.monotonic()
                        if now - last_emit < self.minimum_emit_interval:
                            continue
                        # Own the data before Vimba recycles its frame buffer.
                        array = frame.as_numpy_ndarray()[:, :, 0].copy()
                        self.frame_ready.emit(array)
                        last_emit = now
        except Exception as exc:
            self.error.emit(f"Camera acquisition stopped: {exc}")
        finally:
            self.finished.emit()

    def request_stop(self):
        self._stop_requested.set()


class AnalysisWorker(QtCore.QObject):
    """Run image fitting and region analysis on a dedicated thread."""

    result_ready = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot(object)
    def analyze(self, frame):
        try:
            self.result_ready.emit(analyze_frame(frame))
        except Exception as exc:
            self.error.emit(f"Frame analysis failed: {exc}")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class Viewer(DesignerDisplay, QtWidgets.QWidget):
    filename = "viewer.ui"
    analyze_requested = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        # self. add_menubar(self.ImageWidget)
        self.left_plot = MplCanvas(self, width=2, height=5)
        self.left_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.top_plot = MplCanvas(self, width=5, height=2)
        self.top_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.pixel_um_conversion = 2.2
        self.left_plot.axes.invert_xaxis()
        self.left_plot.axes.invert_yaxis()
        self.LeftPlotLayout.addWidget(self.left_plot)
        self.TopPlotLayout.addWidget(self.top_plot)
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
        self.RefGroupBox.setVisible(False)
        self._start_workers()

    def _start_workers(self):
        self.analysis_thread = QtCore.QThread(self)
        self.analysis_worker = AnalysisWorker()
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analyze_requested.connect(self.analysis_worker.analyze)
        self.analysis_worker.result_ready.connect(self._render_result)
        self.analysis_worker.error.connect(self._analysis_failed)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.start()

        self.acquisition_thread = QtCore.QThread(self)
        self.acquisition_worker = AcquisitionWorker(max_update_hz=30)
        self.acquisition_worker.moveToThread(self.acquisition_thread)
        self.acquisition_thread.started.connect(self.acquisition_worker.run)
        self.acquisition_worker.frame_ready.connect(self._queue_frame)
        self.acquisition_worker.error.connect(self._show_error)
        self.acquisition_worker.finished.connect(self.acquisition_thread.quit)
        self.acquisition_worker.finished.connect(self.acquisition_worker.deleteLater)
        self.acquisition_thread.start()

    @QtCore.pyqtSlot(object)
    def _queue_frame(self, frame):
        if self._stopping:
            return
        if self._analysis_busy:
            self._pending_frame = frame
            return
        self._analysis_busy = True
        self.analyze_requested.emit(frame)

    @QtCore.pyqtSlot(object)
    def _render_result(self, result):
        if self._stopping:
            return

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
        self.left_plot.axes.set_ylim(len(result.row_pixel) - 1, 0)
        self.top_plot.axes.set_xlim(0, len(result.column_pixel) - 1)
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
        if result.column_width is not None:
            self.wxPixelLabel.setText(f"{result.column_width:.2f}")
            self.wxmmLabel.setText(
                f"{result.column_width * self.pixel_um_conversion / 1000:.2f}"
            )

    def _set_region_labels(self, result):
        if result.centroid is None:
            self.curve.setData([], [])
            self.crosshair.setData(pos=[])
            self.CentroidLabel.setText("No foreground detected")
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
        frame = self._pending_frame
        self._pending_frame = None
        self._analysis_busy = True
        self.analyze_requested.emit(frame)

    @QtCore.pyqtSlot(str)
    def _show_error(self, message):
        self.MiscLabel.setText(message)

    def closeEvent(self, event):
        self._stopping = True
        self._pending_frame = None
        if self.acquisition_thread.isRunning():
            self.acquisition_worker.request_stop()
        self.acquisition_thread.quit()
        self.analysis_thread.quit()
        self.acquisition_thread.wait(5000)
        self.analysis_thread.wait(5000)
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
