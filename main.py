from PyQt5 import QtWidgets, QtCore, QtGui
import pydm
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


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class Viewer(DesignerDisplay, QtWidgets.QWidget):
    filename = "viewer.ui"

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
        self.curve = None
        self.crosshair = None
        self.ellipse_pen = pg.mkPen(color="w", width=3)
        self.t = np.linspace(0, 2 * np.pi, 100)
        self.vmb = vm.VmbSystem.get_instance()
        with self.vmb:
            self.cam = self.vmb.get_all_cameras()[0]
            with self.cam:
                frame = self.cam.get_frame()
                arr = frame.as_numpy_ndarray()[:, :, 0]
        self.ImageView.setImage(arr)
        sumx = np.sum(arr, 1)
        sumy = np.sum(arr, 0)
        self.x_pixel = range(len(sumx))
        self.y_pixel = range(len(sumy))
        self.left_plot_data = self.left_plot.axes.plot(sumx, self.x_pixel)[0]
        self.top_plot_data = self.top_plot.axes.plot(self.y_pixel, sumy)[0]
        self.left_plot_fit = self.left_plot.axes.plot(sumx, self.x_pixel)[0]
        self.top_plot_fit = self.top_plot.axes.plot(sumy)[0]
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_cb)
        self.RefGroupBox.setVisible(False)
        self.update_cb()

    def update_cb(self):
        """
        callback method to update viewer image
        """

        # Get image array from camera and send to viewer
        with self.vmb:
            with self.cam:
                frame = self.cam.get_frame()
                arr = frame.as_numpy_ndarray()[:, :, 0]
        self.ImageView.setImage(arr)

        # Gaussian fit plots and calculations
        sumx = np.sum(arr, 1)
        sumy = np.sum(arr, 0)
        self.left_plot_data.set_data(sumx, self.x_pixel)
        self.top_plot_data.set_data(self.y_pixel, sumy)
        try:
            left_parameters, _ = curve_fit(gaussian_fit, self.x_pixel, sumx)
            self.left_plot_fit.set_xdata(
                gaussian_fit(
                    self.x_pixel,
                    left_parameters[0],
                    left_parameters[1],
                    left_parameters[2],
                )
            )
            self.wyPixelLabel.setText(f"{left_parameters[1]:.2f}")
            self.wymmLabel.setText(
                f"{left_parameters[1]*self.pixel_um_conversion/1000:.2f}"
            )
        except RuntimeError:
            pass
        try:
            top_parameters, _ = curve_fit(gaussian_fit, self.y_pixel, sumy)
            self.top_plot_fit.set_ydata(
                gaussian_fit(
                    self.y_pixel,
                    top_parameters[0],
                    top_parameters[1],
                    top_parameters[2],
                )
            )
            self.wxPixelLabel.setText(f"{top_parameters[1]:.2f}")
            self.wxmmLabel.setText(
                f"{top_parameters[1]*self.pixel_um_conversion/1000:.2f}"
            )
        except RuntimeError:
            pass
        self.left_plot.draw_idle()
        self.top_plot.draw_idle()

        # Ellipse outline and centroid calculation and display
        threshold_val = filters.threshold_otsu(arr)
        labeled_foreground = (arr > threshold_val).astype(int)
        properties = regionprops(labeled_foreground, arr)
        com_x, com_y = (
            properties[0].weighted_centroid[1],
            properties[0].weighted_centroid[0],
        )
        orientation = -1 * (properties[0].orientation + (np.pi / 2))
        semimajor = properties[0].axis_major_length / 2
        semiminor = properties[0].axis_minor_length / 2
        Ell = np.array(
            [
                semimajor * np.cos(self.t),
                semiminor * np.sin(self.t),
            ]
        )
        Ell_rot = np.zeros((2, Ell.shape[1]))
        t_rot = orientation
        R_rot = np.array(
            [
                [np.cos(t_rot), -1 * np.sin(t_rot)],
                [np.sin(t_rot), np.cos(t_rot)],
            ]
        )
        for i in range(Ell.shape[1]):
            Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
        x = com_x + Ell_rot[0, :]
        y = com_y + Ell_rot[1, :]
        if self.curve is not None:
            self.ImageView.removeItem(self.curve)
        if self.crosshair is not None:
            self.ImageView.removeItem(self.crosshair)
        self.curve = pg.PlotCurveItem(x=x, y=y, pen=self.ellipse_pen)
        self.crosshair = pg.ScatterPlotItem(
            pos=[(com_x, com_y)],
            symbol="+",
            pen=self.ellipse_pen,
        )
        self.ImageView.addItem(self.curve)
        self.ImageView.addItem(self.crosshair)
        self.CentroidLabel.setText(
            f"Pixel ({com_x:.2f}, {com_y:.2f}); mm ({com_x*self.pixel_um_conversion/1000:.2f}, {com_y*self.pixel_um_conversion/1000:.2f})"
        )
        self.SemimajorLabel.setText(
            f"Pixel: {semimajor:.2f}; mm: {semimajor*self.pixel_um_conversion/1000:.2f}"
        )
        self.SemiminorLabel.setText(
            f"Pixel: {semiminor:.2f}; mm: {semiminor*self.pixel_um_conversion/1000:.2f}"
        )
        self.OrientationLabel.setText(f"{orientation:.2f}")
        self.timer.start(10)

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
