from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap 
from PIL import Image
import module_irm_new

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setStyleSheet("background-color: #333333;")
        self.setWindowTitle("Image Compression")

        # Create a label to display the text
        self.label = QLabel("Image Extension", self)
        font = self.label.font()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setStyleSheet("color: gray; font-weight: bold;")
        self.label.setAlignment(Qt.AlignHCenter)

        # Set the label as the central widget
        self.setCentralWidget(self.label)

        # Set the window size
        self.resize(700, 500)

        # Create a widget to contain buttons and text fields
        main_widget = QWidget()

        # Create a vertical layout for buttons and text fields
        main_layout = QVBoxLayout(main_widget)
        main_layout.setAlignment(Qt.AlignTop)

        # Create labels and text fields for "Taux" and "MSE"
        taux_label = QLabel("Taux", self)
        taux_label.setStyleSheet("color: white; font-weight: bold;")
        mse_label = QLabel("MSE", self)
        mse_label.setStyleSheet("color: white; font-weight: bold;")

        self.taux_text = QLineEdit(self)
        self.taux_text.setStyleSheet("background-color: gray; color: white;")
        self.mse_text = QLineEdit(self)
        self.mse_text.setStyleSheet("background-color: gray; color: white;")

        # Create a gray frame to enclose the labels
        frame_style = "border: 2px solid gray; border-radius: 5px; background-color: #777777;"
        path_frame = QWidget()
        path_frame.setStyleSheet(frame_style)
        path_layout = QVBoxLayout(path_frame)
        # Create a frame to enclose the labels
        frame_style = "border: 2px solid gray; border-radius: 5px; background-color: #777777;"
        taux_frame = QWidget()
        taux_frame.setStyleSheet(frame_style)
        taux_layout = QVBoxLayout(taux_frame)
        taux_layout.addWidget(taux_label)
        taux_layout.addWidget(self.taux_text)

        mse_frame = QWidget()
        mse_frame.setStyleSheet(frame_style)
        mse_layout = QVBoxLayout(mse_frame)
        mse_layout.addWidget(mse_label)
        mse_layout.addWidget(self.mse_text)

        # Create a layout to hold the "Taux" and "MSE" labels and text fields
        labels_layout = QHBoxLayout()
        labels_layout.addWidget(taux_frame)
        labels_layout.addWidget(mse_frame)
        main_layout.addLayout(labels_layout)

        # Add the label for the image path
        path_label = QLabel("Image Path:", self)
        path_label.setStyleSheet("color: white;")
        path_layout.addWidget(path_label)

        # Text field to display the image path
        self.image_path_field = QLineEdit(self)
        self.image_path_field.setReadOnly(True)
        self.image_path_field.setStyleSheet("background-color: gray; color: white;")  
        path_layout.addWidget(self.image_path_field)

        # Add the path frame to the main window
        main_layout.addWidget(path_frame)

        # Create a gray frame to enclose the labels
        name_frame = QWidget()
        name_frame.setStyleSheet(frame_style)
        name_layout = QVBoxLayout(name_frame)

        # Add the label for the image name
        name_label = QLabel("Image Name:", self)
        name_label.setStyleSheet("color: white;")
        name_layout.addWidget(name_label)

        # Text field to display the image name
        self.image_name_field = QLineEdit(self)
        self.image_name_field.setReadOnly(True)
        self.image_name_field.setStyleSheet("background-color: gray; color: white;")  
        name_layout.addWidget(self.image_name_field)

        # Add the name frame to the main window
        main_layout.addWidget(name_frame)

        # Create a widget to contain buttons
        button_widget = QWidget()

        # Create a vertical layout for buttons
        button_layout = QVBoxLayout(button_widget)
        button_layout.setAlignment(Qt.AlignTop)

        # Button to choose a file
        choose_button = QPushButton("Browse")
        choose_button.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #AAAAAA); color: white; border: none; padding: 10px; text-align: center; text-decoration: none; font-size: 16px; margin: 4px 2px; border-radius: 8px;")
        button_layout.addWidget(choose_button)

        # Button to display image information
        info_button = QPushButton("Information")
        info_button.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #AAAAAA); color: white; border: none; padding: 10px; text-align: center; text-decoration: none; font-size: 16px; margin: 4px 2px; border-radius: 8px;")
        button_layout.addWidget(info_button)

        # Add the button layout to the main window
        main_layout.addWidget(button_widget)


        # Create a horizontal layout for compression and decompression buttons
        compress_decompress_layout = QHBoxLayout()

        # Button for compression
        compress_button = QPushButton("Start Compression")
        compress_button.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #AAAAAA); color: white; border: none; padding: 10px; text-align: center; text-decoration: none; font-size: 16px; margin: 4px 2px; border-radius: 8px;")
        compress_decompress_layout.addWidget(compress_button)

        # Button for decompression
        decompress_button = QPushButton("Show image.irm")
        decompress_button.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #AAAAAA); color: white; border: none; padding: 10px; text-align: center; text-decoration: none; font-size: 16px; margin: 4px 2px; border-radius: 8px;")
        compress_decompress_layout.addWidget(decompress_button)

        # Add the horizontal layout to the vertical layout
        button_layout.addLayout(compress_decompress_layout)

        # Add the button widget to the main window
        main_layout.addWidget(button_widget)

        # Connect button signals to slots
        choose_button.clicked.connect(self.choose_image_file)
        compress_button.clicked.connect(self.compress_image)
        decompress_button.clicked.connect(self.decompress_image)
        info_button.clicked.connect(self.show_image_info)

        # Create a layout for "Original image" and "IRM Image" labels
        label_layout = QHBoxLayout()
        self.processed_image_path_label = QLabel("Original image", self)  # Define the processed_image_path_label
        self.processed_image_path_label.setStyleSheet("color: white; font-weight: bold; font-size: 18px;margin-left: 100px;")
        self.irm_label = QLabel("IRM Image", self)
        self.irm_label.setStyleSheet("color: white; font-weight: bold; font-size: 18px;margin-left: 75px;")
        label_layout.addWidget(self.processed_image_path_label)
        label_layout.addSpacing(30)
        label_layout.addWidget(self.irm_label)

        # Add the label layout to the main layout
        main_layout.addLayout(label_layout)

        # Set the main widget
        self.setCentralWidget(main_widget)


        self.twoimages = QHBoxLayout()
        # Create a label to display the image ORIGINAL
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet('margin-left: 20px')
        self.twoimages.addWidget(self.image_label)


        # Create a label to display the image IRM
        self.irm_label = QLabel(self)
        self.irm_label.setStyleSheet('margin-left: 20px')
        self.twoimages.addWidget(self.irm_label)
        main_layout.addLayout(self.twoimages)


        # Créez un bouton Clear en bas de l'interface
        clear_button = QPushButton("Clear")
        clear_button.setStyleSheet("background-color: white; color: black; font-weight: bold;")
        button_layout.addWidget(clear_button)

        # Connectez le signal clicked du bouton à la méthode clear_image_and_fields
        clear_button.clicked.connect(self.clear_image_and_fields)

        # Ajoutez un peu d'espace entre le bouton Clear et le bas de l'interface
        main_layout.addSpacing(20)

    def choose_image_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose an image", "", "Image (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_name:
            self.image_name_field.setText(file_name.split('/')[-1])  # Display only the file name
            self.image_path_field.setText(file_name)  # Display the full file path

            # Display the image
            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaledToWidth(300)  # Resize the image for display
            self.image_label.setPixmap(pixmap)

    def compress_image(self):
        self.file_path = self.image_path_field.text()
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
        try:
            self.compressed_image_path = f"{self.file_path.split('.')[0]}_compressed.irm"
            save_options = QFileDialog.Options()
            save_options |= QFileDialog.DontUseNativeDialog
            output_path, _ = QFileDialog.getSaveFileName(self, "Save Decompressed Image", "", "JPEG Files (*.jpeg);;PNG Files (*.png)", options=save_options)
            if output_path:
                self.decompressed_image_path = output_path
            head = module_irm_new.compression_irm(self.file_path, self.compressed_image_path)
            self.image_dec =  module_irm_new.decompression_irm(head,self.compressed_image_path,self.decompressed_image_path)
            self.image_dec.save(self.decompressed_image_path)
            taux = module_irm_new.compression_ratio(self.file_path, self.decompressed_image_path)
            mse = module_irm_new.MSE(self.file_path, self.decompressed_image_path)
            self.mse_text.setText(str(mse))
            self.taux_text.setText(str(taux))
            # Afficher un message lorsque la compression est terminée
            QMessageBox.information(self, "Compression Complete", "Image compression is complete.")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error compressing the image: {str(e)}")
        
    def decompress_image(self):
        if not hasattr(self, 'decompressed_image_path') or not self.decompressed_image_path:
            QMessageBox.warning(self, "Warning", "Please compress the image first.")
            return
    
        pixmap = QPixmap(self.decompressed_image_path)
        pixmap = pixmap.scaledToWidth(300) 
        self.irm_label.setPixmap(pixmap)

    def show_image_info(self):
        file_path = self.image_path_field.text()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return

        try:
            self.image = Image.open(file_path)
            info_str = f"Size: {self.image.size}\nFormat: {self.image.format}\nMode: {self.image.mode}"
            QMessageBox.information(self, "Image Information", info_str)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Unable to open the image: {str(e)}")

    def clear_image_and_fields(self):
        # Effacez l'image affichée
        self.image_label.clear()
        self.irm_label.clear()
        # Réinitialisez les champs de texte
        self.image_path_field.setText("")
        self.image_name_field.setText("")


app = QApplication([])

window = MainWindow()
window.show()

app.exec_()
