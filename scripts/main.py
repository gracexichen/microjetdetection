# imports
from PyQt5.QtWidgets import *
from PyQt5 import uic

from graphs import *
from frame import *
from video import *


class MyGUI(QMainWindow):
    """Main UI window extended from QMainWindow

    Main UI window for the application. Contains all the widgets and processes any input
    """

    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi("mainwindow.ui", self)

        self.conversion_factor = 0.000038
        self.video = Video()
        self.graphs = Graphs(self, width=5, height=4, dpi=100)
        self.threshold_image = None
        self.line1 = []
        self.line2 = []
        self.line3 = []
        self.line4 = []

        self.graphingwidget.addWidget(self.graphs)
        self.graphs.setMinimumSize(QSize(400, 400))
        self.video.image_update.connect(self.image_update_slot)
        self.video.threshold_image_update.connect(self.threshold_update_image)
        self.video.info_update.connect(self.update_info)
        self.video.graph_update.connect(self.update_graph)
        self.video.slider_update.connect(self.update_slider)
        self.video_slider.sliderMoved.connect(self.change_slider_value)
        self.pushButton.clicked.connect(self.pause_play_video)
        self.pushButton_2.clicked.connect(self.upload_video)
        self.settings_button.clicked.connect(self.show_settings)
        self.detection_outline_button.clicked.connect(self.set_outline)
        self.horizontalSlider.valueChanged.connect(self.speed_change)
        self.detection_outline_button.setVisible(False)
        self.horizontalSlider.setVisible(False)
        self.speed_label.setVisible(False)
        self.set_threshold_button.clicked.connect(self.start_video)
        self.visible = True
        self.video.video_status.connect(self.update_video_status)
        self.threshold_scroll_bar.valueChanged.connect(
            self.update_threshold_value)
        self.show()

    def start_video(self):
        self.video.threshold = self.threshold_scroll_bar.value()
        Frame.frame_rate = int(self.frame_rate_input.text())
        self.frame_rate_input.setVisible(False)
        self.threshold_scroll_bar.setVisible(False)
        self.set_threshold_button.setVisible(False)
        self.video.start()

    def update_threshold_value(self, new_value):
        ret, thresh1 = cv.threshold(
            self.threshold_image, new_value, 255, cv.THRESH_BINARY_INV)
        converted_image = QImage(
            thresh1.data, thresh1.shape[1], thresh1.shape[0], QImage.Format_RGB888)
        self.videolabel.setPixmap(QPixmap.fromImage(
            converted_image.scaled(512, 512, Qt.KeepAspectRatio)))

    def threshold_update_image(self, unconverted_img, new_img):
        self.threshold_image = unconverted_img
        self.videolabel.setPixmap(QPixmap.fromImage(new_img))

    def update_video_status(self, status):
        self.videolabel.clear()
        self.videolabel.setText(status)

    def update_slider(self, frame_num):
        """Updates the slider with the current frame number

        Called to update the slider with the progress of the video as it plays. If the frame number
        reaches the maximum count of frames in the video, the scroll bar resets to 0 as is also the
        video.

        Args:
        frame_num: the frame number to update the slider to

        """
        if self.video_slider.maximum() == 0:
            self.video_slider.setMaximum(self.video.max_frames)
        self.video_slider.setValue(frame_num)

    def change_slider_value(self, new_frame_num):
        """Updates the frame number the video is playing when slider is moved

        Sets frame number of video to the new frame number the scroll bar is moved to

        Args:
        new_frame_num: frame number to set the video to
        """
        self.video.frame_number = new_frame_num
        Graphs.frameData = []
        Graphs.radiusData = []
        Graphs.lengthData = []
        Graphs.widthData = []
        Graphs.speedData = []

    def speed_change(self, multiplier):
        """Changes speed the video is played

        Sets the speed to the value of the slider after being moved multiplied by 100

        Args:
        multiplier: the value of the scrollbar
        """
        self.video.speed = (11 - multiplier) * 100

    def set_outline(self):
        """ Toggles whether the detection outline of the jet and droplets should be displayed on the image
        """
        Frame.outline = not Frame.outline

    def show_settings(self):
        """ Toggles between showing and hiding the settings when the setting button is pressed
        """
        self.detection_outline_button.setVisible(self.visible)
        self.horizontalSlider.setVisible(self.visible)
        self.speed_label.setVisible(self.visible)
        self.visible = not self.visible

    def image_update_slot(self, new_img):
        """ Updates the image displayed for the video with new image (next frame)

        Args:
        new_img: new frame of the video
        """
        self.videolabel.setPixmap(QPixmap.fromImage(new_img))

    def pause_play_video(self):
        """Toggles video between pause and play, adapts the text as well
        """
        if self.video.is_paused:
            self.video.play()
            self.pushButton.setText("Pause")
        else:
            self.video.pause()
            self.pushButton.setText("Play")

    def upload_video(self):
        """Prompts for file upload and plays video if loaded
        """
        self.video.upload()

    def update_info(self, radius, length, width, speed):
        """Update the labels for each properties with new data for new frame

        Args:
        radius: average radius of the droplets
        length: length of the jet
        width: width of the jet
        speed: speed of the droplets
        """
        self.label.setText(str(self.video.frame_number))
        self.droplet_radius_label.setText(
            "Droplet radius: " + self.scientific_notation(radius * self.conversion_factor) + "m")
        self.jet_length_label.setText(
            "Jet Length: " + self.scientific_notation(length * self.conversion_factor) + "m")
        self.jet_width_label.setText(
            "Jet Width: " + self.scientific_notation(width * self.conversion_factor) + "m")
        self.droplet_speed_label.setText(
            "Droplet speed: " + self.scientific_notation(speed * self.conversion_factor) + "m" + "/" + "s")

    def scientific_notation(self, number):
        num_str = str(number)
        num_length = len(num_str.replace('.', '').lstrip('0'))

        if num_length >= 5:
            return f"{number:.2e}"
        else:
            return str(number)

    def update_graph(self, radius, length, width, speed, frame):
        """Update the graph with the new data for the frame

        Clears the current graph and adds data to array
        If the arrays containing the properties exceeds 10 values, remove the value fron the front
        Plots data and also rescales axis

        Args:
        radius: average radius of the droplets
        length: length of the jet
        width: width of the jet
        speed: speed of the droplets
        frame: the frame number video is on
        """
        radius = radius * self.conversion_factor
        length = length * self.conversion_factor
        width = width * self.conversion_factor
        speed = speed * self.conversion_factor

        try:
            l1 = self.line1.pop()
            l2 = self.line2.pop()
            l3 = self.line3.pop()
            l4 = self.line4.pop()

            l1.remove()
            l2.remove()
            l3.remove()
            l4.remove()
        except:
            print("Lines not clearing correctly")

        Graphs.frameData.append(frame)
        Graphs.radiusData.append(radius)
        Graphs.lengthData.append(length)
        Graphs.widthData.append(width)
        Graphs.speedData.append(speed)

        if len(self.graphs.frameData) > 10:
            Graphs.frameData.pop(0)
            Graphs.radiusData.pop(0)
            Graphs.lengthData.pop(0)
            Graphs.widthData.pop(0)
            Graphs.speedData.pop(0)

        self.line1 = self.graphs.ax1.plot(Graphs.frameData, Graphs.radiusData,
                                          marker=".", markersize=3, markeredgecolor="red")
        self.line2 = self.graphs.ax2.plot(Graphs.frameData, Graphs.speedData,
                                          marker=".", markersize=3, markeredgecolor="red")
        self.line3 = self.graphs.ax3.plot(Graphs.frameData, Graphs.widthData,
                                          marker=".", markersize=3, markeredgecolor="red")
        self.line4 = self.graphs.ax4.plot(Graphs.frameData, Graphs.lengthData,
                                          marker=".", markersize=3, markeredgecolor="red")

        self.graphs.ax1.relim()
        self.graphs.ax2.relim()
        self.graphs.ax3.relim()
        self.graphs.ax4.relim()

        self.graphs.ax1.autoscale_view()
        self.graphs.ax2.autoscale_view()
        self.graphs.ax3.autoscale_view()
        self.graphs.ax4.autoscale_view()

        self.graphs.draw_idle()

# start application


def main():
    app = QApplication([])
    window = MyGUI()
    app.exec()


if __name__ == '__main__':
    main()
