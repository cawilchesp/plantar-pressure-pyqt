"""
Frontend

This file contains main UI class and methods to control components operations.
"""

from PyQt6 import QtGui, QtWidgets, QtCore
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import QSettings, Qt

import sys
import pandas as pd
from pathlib import Path

import material3_components as mt3
import backend
import patient
import database


class App(QWidget):
    def __init__(self):
        """ UI main application """
        super().__init__()
        # --------
        # Settings
        # --------
        self.settings = QSettings(f'{sys.path[0]}/settings.ini', QSettings.Format.IniFormat)
        self.language_value = int(self.settings.value('language'))
        self.theme_value = eval(self.settings.value('theme'))

        self.idioma_dict = {0: ('ESP', 'SPA'), 1: ('ING', 'ENG')}
    
        # ---------
        # Variables
        # ---------
        self.patient_data = None
        self.data_lat_max = 0.0
        self.data_lat_t_max = 0.0
        self.data_lat_min = 0.0
        self.data_lat_t_min = 0.0
        self.data_ap_max = 0.0
        self.data_ap_t_max = 0.0
        self.data_ap_min = 0.0
        self.data_ap_t_min = 0.0
        self.lat_text_1 = None
        self.lat_text_2 = None
        self.ap_text_1 = None
        self.ap_text_2 = None

        # ----------------
        # Generación de UI
        # ----------------
        width = 1300
        height = 700
        screen_x = int(self.screen().availableGeometry().width() / 2 - (width / 2))
        screen_y = int(self.screen().availableGeometry().height() / 2 - (height / 2))

        if self.language_value == 0:
            self.setWindowTitle('Presión Plantar')
        elif self.language_value == 1:
            self.setWindowTitle('Plantar Pressure')
        self.setGeometry(screen_x, screen_y, width, height)
        self.setMinimumSize(1300, 700)
        if self.theme_value:
            self.setStyleSheet(f'QWidget {{ background-color: #E5E9F0; color: #000000 }}'
                f'QComboBox QListView {{ border: 1px solid #000000; border-radius: 4;'
                f'background-color: #B2B2B2; color: #000000 }}')
        else:
            self.setStyleSheet(f'QWidget {{ background-color: #3B4253; color: #E5E9F0 }}'
                f'QComboBox QListView {{ border: 1px solid #E5E9F0; border-radius: 4;'
                f'background-color: #2E3441; color: #E5E9F0 }}')
        
        # -----------
        # Card Título
        # -----------
        self.titulo_card = mt3.Card(self, 'titulo_card',
            (8, 8, width-16, 48), ('',''), self.theme_value, self.language_value)


        # Espacio para título de la aplicación, logo, etc.

        
        self.idioma_menu = mt3.Menu(self.titulo_card, 'idioma_menu',
            (8, 8, 72), 2, 2, self.idioma_dict, self.theme_value, self.language_value)
        self.idioma_menu.setCurrentIndex(self.language_value)
        self.idioma_menu.currentIndexChanged.connect(self.on_idioma_menu_currentIndexChanged)
        
        self.tema_switch = mt3.Switch(self.titulo_card, 'tema_switch',
            (8, 8, 48), ('', ''), ('light_mode.png','dark_mode.png'), 
            self.theme_value, self.theme_value, self.language_value)
        self.tema_switch.clicked.connect(self.on_tema_switch_clicked)

        self.database_button = mt3.IconButton(self.titulo_card, 'database_button',
            (8, 8), 'database.png', self.theme_value)
        self.database_button.clicked.connect(self.on_database_button_clicked)

        self.manual_button = mt3.IconButton(self.titulo_card, 'manual_button',
            (8, 8), 'help.png', self.theme_value)
        self.manual_button.clicked.connect(self.on_manual_button_clicked)

        self.about_button = mt3.IconButton(self.titulo_card, 'about_button',
            (8, 8), 'mail_L.png', self.theme_value)
        self.about_button.clicked.connect(self.on_about_button_clicked)

        self.aboutQt_button = mt3.IconButton(self.titulo_card, 'aboutQt_button',
            (8, 8), 'about_qt.png', self.theme_value)
        self.aboutQt_button.clicked.connect(self.on_aboutQt_button_clicked)

        # -------------
        # Card Paciente
        # -------------
        self.paciente_card = mt3.Card(self, 'paciente_card',
            (8, 64, 180, 128), ('Paciente', 'Patient'), 
            self.theme_value, self.language_value)
        
        y_1 = 48
        self.pacientes_menu = mt3.Menu(self.paciente_card, 'pacientes_menu',
            (8, y_1, 164), 10, 10, {}, self.theme_value, self.language_value)
        # self.pacientes_menu.textActivated.connect(self.on_pacientes_menu_textActivated)

        y_1 += 40
        self.paciente_add_button = mt3.IconButton(self.paciente_card, 'paciente_add_button',
            (60, y_1), 'person_add.png', self.theme_value)
        self.paciente_add_button.clicked.connect(self.on_paciente_add_button_clicked)

        self.paciente_edit_button = mt3.IconButton(self.paciente_card, 'paciente_edit_button',
            (100, y_1), 'edit.png', self.theme_value)
        # self.paciente_edit_button.clicked.connect(self.on_paciente_edit_button_clicked)

        self.paciente_del_button = mt3.IconButton(self.paciente_card, 'paciente_del_button',
            (140, y_1), 'person_off.png', self.theme_value)
        # self.paciente_del_button.clicked.connect(self.on_paciente_del_button_clicked)

        # -------------
        # Card Análisis
        # -------------
        self.analisis_card = mt3.Card(self, 'analisis_card',
            (8, 200, 180, 128), ('Análsis', 'Analysis'), 
            self.theme_value, self.language_value)

        y_2 = 48
        self.analisis_menu = mt3.Menu(self.analisis_card, 'analisis_menu',
            (8, y_2, 164), 10, 10, {}, self.theme_value, self.language_value)
        self.analisis_menu.setEnabled(False)
        # self.analisis_menu.textActivated.connect(self.on_analisis_menu_textActivated)

        y_2 += 40
        self.analisis_add_button = mt3.IconButton(self.analisis_card, 'analisis_add_button',
            (100, y_2), 'new.png', self.theme_value)
        self.analisis_add_button.setEnabled(False)
        # self.analisis_add_button.clicked.connect(self.on_analisis_add_button_clicked)

        self.analisis_del_button = mt3.IconButton(self.analisis_card, 'analisis_del_button',
            (140, y_2), 'delete.png', self.theme_value)
        self.analisis_del_button.setEnabled(False)
        # self.analisis_del_button.clicked.connect(self.on_analisis_del_button_clicked)

        # ----------------
        # Card Información
        # ----------------
        self.info_card = mt3.Card(self, 'info_card',
            (8, 336, 180, 312), ('Información', 'Information'), 
            self.theme_value, self.language_value)
        
        y_3 = 48
        self.apellido_value = mt3.ValueLabel(self.info_card, 'apellido_value',
            (8, y_3, 164), self.theme_value)

        y_3 += 32
        self.nombre_value = mt3.ValueLabel(self.info_card, 'nombre_value',
            (8, y_3, 164), self.theme_value)

        y_3 += 32
        self.id_label = mt3.IconLabel(self.info_card, 'id_label',
            (8, y_3), 'id', self.theme_value)

        self.id_value = mt3.ValueLabel(self.info_card, 'id_value',
            (48, y_3, 124), self.theme_value)

        y_3 += 32
        self.fecha_label = mt3.IconLabel(self.info_card, 'fecha_label',
            (8, y_3), 'calendar', self.theme_value)

        self.fecha_value = mt3.ValueLabel(self.info_card, 'fecha_value',
            (48, y_3, 124), self.theme_value)
        
        y_3 += 32
        self.sex_label = mt3.IconLabel(self.info_card, 'sex_label',
            (8, y_3), '', self.theme_value)

        self.sex_value = mt3.ValueLabel(self.info_card, 'sex_value',
            (48, y_3, 124), self.theme_value)

        y_3 += 32
        self.peso_label = mt3.IconLabel(self.info_card, 'peso_label',
            (8, y_3), 'weight', self.theme_value)

        self.peso_value = mt3.ValueLabel(self.info_card, 'peso_value',
            (48, y_3, 124), self.theme_value)

        y_3 += 32
        self.altura_label = mt3.IconLabel(self.info_card, 'altura_label',
            (8, y_3), 'height', self.theme_value)

        self.altura_value = mt3.ValueLabel(self.info_card, 'altura_value',
            (48, y_3, 124), self.theme_value)

        y_3 += 32
        self.bmi_value = mt3.ValueLabel(self.info_card, 'bmi_value',
            (48, y_3, 124), self.theme_value)
        
        # -----------------
        # Cards Main Window
        # -----------------
        self.presion_plot_card = mt3.Card(self, 'presion_plot_card',
            (196, 64, 900, 215), ('Mapa de Presiones Plantares','Plantar Pressures Map'), 
            self.theme_value, self.language_value)
    #     self.somatotipo_plot = backend.MPLCanvas(self.presion_plot_card, self.theme_value)

        # -------------
        # Card Opciones
        # -------------
        self.opciones_card = mt3.Card(self, 'opciones_card',
            (196, 64, 228, 168), ('Opciones', 'Options'), 
            self.theme_value, self.language_value)

        y_8 = 48
        self.opciones_plot_label = mt3.ItemLabel(self.opciones_card, 'opciones_plot_label',
            (8, y_8), ('Opciones de la gráfica', 'Plot Options'), self.theme_value, self.language_value)

        y_8 += 20
        self.regiones_chip = mt3.Chip(self.opciones_card, 'regiones_chip',
            (8, y_8, 124), ('Regiones', 'Regions'), ('done.png','none.png'), 
            False, self.theme_value, self.language_value)
        # self.regiones_chip.clicked.connect(self.on_regiones_chip_clicked)

        self.presion_pico_local_chip = mt3.Chip(self.opciones_card, 'presion_pico_local_chip',
            (140, y_8, 168), ('Presiones Pico Locales', 'Local Peak Pressures'), ('done.png','none.png'), 
            False, self.theme_value, self.language_value)
        # self.presion_pico_local_chip.clicked.connect(self.on_presion_pico_local_chip_clicked)

        self.baricentros_chip = mt3.Chip(self.opciones_card, 'baricentros_chip',
            (316, y_8, 124), ('Baricentros', 'Barycenters'), ('done.png','none.png'), 
            False, self.theme_value, self.language_value)
        # self.baricentros_chip.clicked.connect(self.on_baricentros_chip_clicked)

        self.distribucion_chip = mt3.Chip(self.opciones_card, 'distribucion_chip',
            (448, y_8, 124), ('Distribuciones', 'Distributions'), ('done.png','none.png'), 
            False, self.theme_value, self.language_value)
        # self.distribucion_chip.clicked.connect(self.on_distribucion_chip_clicked)

        y_8 += 40
        self.opciones_results_label = mt3.ItemLabel(self.opciones_card, 'opciones_results_label',
            (8, y_8), ('Opciones de los resultados', 'Results Options'), self.theme_value, self.language_value)
        
        y_8 += 20
        self.picos_button = mt3.SegmentedButton(self.opciones_card, 'picos_button',
            (8, y_8, 136), ('Picos de Presión', 'Peak Pressures'), ('done.png','none.png'), 'left', 
            False, self.theme_value, self.language_value)
        # self.picos_button.clicked.connect(self.on_picos_button_clicked)

        self.means_button = mt3.SegmentedButton(self.opciones_card, 'means_button',
            (144, y_8, 144), ('Presiones Medias', 'Mean Pressures'), ('done.png','none.png'), 'center', 
            False, self.theme_value, self.language_value)
        # self.means_button.clicked.connect(self.on_means_button_clicked)

        self.areas_button = mt3.SegmentedButton(self.opciones_card, 'areas_button',
            (288, y_8, 148), ('Áreas de Contacto', 'Contact Areas'), ('done.png','none.png'), 'right', 
            False, self.theme_value, self.language_value)
        # self.areas_button.clicked.connect(self.on_areas_button_clicked)

        # ------------------------
        # Card Parámetros Globales
        # ------------------------
        self.globales_card = mt3.Card(self, 'globales_card',
            (8, 8, 424, 216), ('Parámetros Globales', 'Global Parameters'), 
            self.theme_value, self.language_value)

        y_4 = 48
        self.presion_total_label = mt3.ItemLabel(self.globales_card, 'presion_total_label',
            (8, y_4), ('Presión Total (KPa)', 'Total Pressure (KPa)'), self.theme_value, self.language_value)
        y_4 += 16
        self.presion_total_value = mt3.ValueLabel(self.globales_card, 'presion_total_value',
            (8, y_4, 64), self.theme_value)
        self.presion_total_percent = mt3.ValueLabel(self.globales_card, 'presion_total_percent',
            (80, y_4, 64), self.theme_value)

        y_4 += 40
        self.presion_left_label = mt3.ItemLabel(self.globales_card, 'presion_left_label',
            (8, y_4), ('Presión Pie Izquierdo (KPa)', 'Left Foot Pressure (KPa)'), self.theme_value, self.language_value)
        self.presion_right_label = mt3.ItemLabel(self.globales_card, 'presion_right_label',
            (216, y_4), ('Presión Pie Derecho (KPa)', 'Right Foot Pressure (KPa)'), self.theme_value, self.language_value)

        y_4 += 16
        self.presion_left_value = mt3.ValueLabel(self.globales_card, 'presion_left_value',
            (8, y_4, 64), self.theme_value)
        self.presion_left_percent = mt3.ValueLabel(self.globales_card, 'presion_left_percent',
            (80, y_4, 64), self.theme_value)
        self.presion_right_value = mt3.ValueLabel(self.globales_card, 'presion_right_value',
            (216, y_4, 64), self.theme_value)
        self.presion_right_percent = mt3.ValueLabel(self.globales_card, 'presion_right_percent',
            (296, y_4, 64), self.theme_value)

        y_4 += 40
        self.presion_antepie_label = mt3.ItemLabel(self.globales_card, 'presion_antepie_label',
            (8, y_4), ('Presión Antepié (KPa)', 'Forefoot Pressure (KPa)'), self.theme_value, self.language_value)
        self.presion_retropie_label = mt3.ItemLabel(self.globales_card, 'presion_retropie_label',
            (216, y_4), ('Presión Retropié (KPa)', 'Rarefoot Pressure (KPa)'), self.theme_value, self.language_value)
        
        y_4 += 16
        self.presion_antepie_value = mt3.ValueLabel(self.globales_card, 'presion_antepie_value',
            (8, y_4, 64), self.theme_value)
        self.presion_antepie_percent = mt3.ValueLabel(self.globales_card, 'presion_antepie_percent',
            (80, y_4, 64), self.theme_value)
        self.presion_retropie_value = mt3.ValueLabel(self.globales_card, 'presion_retropie_value',
            (216, y_4, 64), self.theme_value)
        self.presion_retropie_percent = mt3.ValueLabel(self.globales_card, 'presion_retropie_percent',
            (296, y_4, 64), self.theme_value)

        # ----------------------------
        # Card Parámetros por Regiones
        # ----------------------------
        self.regiones_card = mt3.Card(self, 'regiones_card',
            (8, 8, 424, 384), ('Parámetros Regionales', 'Regional Parameters'), 
            self.theme_value, self.language_value)

        y_5 = 48
        self.talon_interno_label = mt3.ItemLabel(self.regiones_card, 'talon_interno_label',
            (8, y_5), ('R1 Talón Interno (KPa)', 'R1 Inside Keel (KPa)'), self.theme_value, self.language_value)
        self.talon_externo_label = mt3.ItemLabel(self.regiones_card, 'talon_externo_label',
            (216, y_5), ('R2 Talón Externo (KPa)', 'R2 Outside Keel (KPa)'), self.theme_value, self.language_value)

        y_5 += 16
        self.left_talon_interno_value = mt3.ValueLabel(self.regiones_card, 'left_talon_interno_value',
            (8, y_5, 64), self.theme_value)
        self.left_talon_interno_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_talon_interno_value.setStyleSheet(f'QLabel#{self.left_talon_interno_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_talon_interno_value = mt3.ValueLabel(self.regiones_card, 'right_talon_interno_value',
            (80, y_5, 64), self.theme_value)
        self.right_talon_interno_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_talon_interno_value.setStyleSheet(f'QLabel#{self.right_talon_interno_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_talon_externo_value = mt3.ValueLabel(self.regiones_card, 'left_talon_externo_value',
            (216, y_5, 64), self.theme_value)
        self.left_talon_externo_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_talon_externo_value.setStyleSheet(f'QLabel#{self.left_talon_externo_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_talon_externo_value = mt3.ValueLabel(self.regiones_card, 'right_talon_externo_value',
            (296, y_5, 64), self.theme_value)
        self.right_talon_externo_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_talon_externo_value.setStyleSheet(f'QLabel#{self.right_talon_externo_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')

        y_5 += 40
        self.mediopie_interno_label = mt3.ItemLabel(self.regiones_card, 'mediopie_interno_label',
            (8, y_5), ('R3 Mediopié Interno (KPa)', 'R3 Inside Midfoot (KPa)'), self.theme_value, self.language_value)
        self.mediopie_externo_label = mt3.ItemLabel(self.regiones_card, 'mediopie_externo_label',
            (216, y_5), ('R4 Mediopié Externo (KPa)', 'R4 Outside Midfoot (KPa)'), self.theme_value, self.language_value)

        y_5 += 16
        self.left_mediopie_interno_value = mt3.ValueLabel(self.regiones_card, 'left_mediopie_interno_value',
            (8, y_5, 64), self.theme_value)
        self.left_mediopie_interno_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_mediopie_interno_value.setStyleSheet(f'QLabel#{self.left_mediopie_interno_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_mediopie_interno_value = mt3.ValueLabel(self.regiones_card, 'right_mediopie_interno_value',
            (80, y_5, 64), self.theme_value)
        self.right_mediopie_interno_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_mediopie_interno_value.setStyleSheet(f'QLabel#{self.right_mediopie_interno_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_mediopie_externo_value = mt3.ValueLabel(self.regiones_card, 'left_mediopie_externo_value',
            (216, y_5, 64), self.theme_value)
        self.left_mediopie_externo_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_mediopie_externo_value.setStyleSheet(f'QLabel#{self.left_mediopie_externo_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_mediopie_externo_value = mt3.ValueLabel(self.regiones_card, 'right_mediopie_externo_value',
            (296, y_5, 64), self.theme_value)
        self.right_mediopie_externo_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_mediopie_externo_value.setStyleSheet(f'QLabel#{self.right_mediopie_externo_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')

        y_5 += 40
        self.metatarsiano_1_label = mt3.ItemLabel(self.regiones_card, 'metatarsiano_1_label',
            (8, y_5), ('R5 1ra Cabeza Metatarsiano (KPa)', 'R5 1st Metatarsal Head (KPa)'), self.theme_value, self.language_value)
        self.metatarsiano_2_label = mt3.ItemLabel(self.regiones_card, 'metatarsiano_2_label',
            (216, y_5), ('R6 2da Cabeza Metatarsiano (KPa)', 'R6 2nd Metatarsal Head (KPa)'), self.theme_value, self.language_value)

        y_5 += 16
        self.left_metatarsiano_1_value = mt3.ValueLabel(self.regiones_card, 'left_metatarsiano_1_value',
            (8, y_5, 64), self.theme_value)
        self.left_metatarsiano_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_1_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_1_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_1_value = mt3.ValueLabel(self.regiones_card, 'right_metatarsiano_1_value',
            (80, y_5, 64), self.theme_value)
        self.right_metatarsiano_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_1_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_1_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_metatarsiano_2_value = mt3.ValueLabel(self.regiones_card, 'left_metatarsiano_2_value',
            (216, y_5, 64), self.theme_value)
        self.left_metatarsiano_2_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_2_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_2_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_2_value = mt3.ValueLabel(self.regiones_card, 'right_metatarsiano_2_value',
            (296, y_5, 64), self.theme_value)
        self.right_metatarsiano_2_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_2_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_2_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')

        y_5 += 40
        self.metatarsiano_3_label = mt3.ItemLabel(self.regiones_card, 'metatarsiano_3_label',
            (8, y_5), ('R7 3ra Cabeza Metatarsiano (KPa)', 'R7 3rd Metatarsal Head (KPa)'), self.theme_value, self.language_value)
        self.metatarsiano_4_label = mt3.ItemLabel(self.regiones_card, 'metatarsiano_4_label',
            (216, y_5), ('R8 4ta Cabeza Metatarsiano (KPa)', 'R8 4th Metatarsal Head (KPa)'), self.theme_value, self.language_value)

        y_5 += 16
        self.left_metatarsiano_3_value = mt3.ValueLabel(self.regiones_card, 'left_metatarsiano_3_value',
            (8, y_5, 64), self.theme_value)
        self.left_metatarsiano_3_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_3_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_3_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_3_value = mt3.ValueLabel(self.regiones_card, 'right_metatarsiano_3_value',
            (80, y_5, 64), self.theme_value)
        self.right_metatarsiano_3_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_3_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_3_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_metatarsiano_4_value = mt3.ValueLabel(self.regiones_card, 'left_metatarsiano_4_value',
            (216, y_5, 64), self.theme_value)
        self.left_metatarsiano_4_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_4_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_4_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_4_value = mt3.ValueLabel(self.regiones_card, 'right_metatarsiano_4_value',
            (296, y_5, 64), self.theme_value)
        self.right_metatarsiano_4_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_4_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_4_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')

        y_5 += 40
        self.metatarsiano_5_label = mt3.ItemLabel(self.regiones_card, 'metatarsiano_5_label',
            (8, y_5), ('R9 5ta Cabeza Metatarsiano (KPa)', 'R9 5th Metatarsal Head (KPa)'), self.theme_value, self.language_value)
        self.dedo_1_label = mt3.ItemLabel(self.regiones_card, 'dedo_1_label',
            (216, y_5), ('R10 1er Dedo (KPa)', 'R10 1st Toe (KPa)'), self.theme_value, self.language_value)

        y_5 += 16
        self.left_metatarsiano_5_value = mt3.ValueLabel(self.regiones_card, 'left_metatarsiano_5_value',
            (8, y_5, 64), self.theme_value)
        self.left_metatarsiano_5_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_5_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_5_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_5_value = mt3.ValueLabel(self.regiones_card, 'right_metatarsiano_5_value',
            (80, y_5, 64), self.theme_value)
        self.right_metatarsiano_5_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_5_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_5_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_dedo_1_value = mt3.ValueLabel(self.regiones_card, 'left_dedo_1_value',
            (216, y_5, 64), self.theme_value)
        self.left_dedo_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_dedo_1_value.setStyleSheet(f'QLabel#{self.left_dedo_1_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_dedo_1_value = mt3.ValueLabel(self.regiones_card, 'right_dedo_1_value',
            (296, y_5, 64), self.theme_value)
        self.right_dedo_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_dedo_1_value.setStyleSheet(f'QLabel#{self.right_dedo_1_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')

        y_5 += 40
        self.dedo_2_label = mt3.ItemLabel(self.regiones_card, 'dedo_2_label',
            (8, y_5), ('R11 2do Dedo (KPa)', 'R11 2nd Toe (KPa)'), self.theme_value, self.language_value)
        self.dedo_3_5_label = mt3.ItemLabel(self.regiones_card, 'dedo_3_5_label',
            (216, y_5), ('R12 3er-5to Dedos (KPa)', 'R12 3rd-5th Toes (KPa)'), self.theme_value, self.language_value)

        y_5 += 16
        self.left_dedo_2_value = mt3.ValueLabel(self.regiones_card, 'left_dedo_2_value',
            (8, y_5, 64), self.theme_value)
        self.left_dedo_2_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_dedo_2_value.setStyleSheet(f'QLabel#{self.left_dedo_2_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_dedo_2_value = mt3.ValueLabel(self.regiones_card, 'right_dedo_2_value',
            (80, y_5, 64), self.theme_value)
        self.right_dedo_2_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_dedo_2_value.setStyleSheet(f'QLabel#{self.right_dedo_2_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_dedo_3_5_value = mt3.ValueLabel(self.regiones_card, 'left_dedo_3_5_value',
            (216, y_5, 64), self.theme_value)
        self.left_dedo_3_5_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_dedo_3_5_value.setStyleSheet(f'QLabel#{self.left_dedo_3_5_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_dedo_3_5_value = mt3.ValueLabel(self.regiones_card, 'right_dedo_3_5_value',
            (296, y_5, 64), self.theme_value)
        self.right_dedo_3_5_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_dedo_3_5_value.setStyleSheet(f'QLabel#{self.right_dedo_3_5_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')

    #     # -------------
    #     # Base de Datos
    #     # -------------
    #     try:
    #         self.patientes_list = backend.create_db('pacientes')
    #         self.estudios_list = backend.create_db('estudios')

    #         for data in self.patientes_list:
    #             self.pacientes_menu.addItem(str(data[4]))
    #         self.pacientes_menu.setCurrentIndex(-1)
    #     except:
    #         self.pacientes_menu.setEnabled(False)
    #         self.paciente_add_button.setEnabled(False)
    #         self.paciente_edit_button.setEnabled(False)
    #         self.paciente_del_button.setEnabled(False)
            
    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.critical(self, 'Error de Base de Datos', 'La base de datos no está configurada')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.critical(self, 'Database Error', 'Database not configured')

    # ----------------
    # Funciones Título
    # ----------------
    def on_idioma_menu_currentIndexChanged(self, index: int) -> None:
        """ Language menu control to change components text language
        
        Parameters
        ----------
        index: int
            Index of language menu control
        
        Returns
        -------
        None
        """
        self.idioma_menu.language_text(index)
        
        self.paciente_card.language_text(index)
        self.analisis_card.language_text(index)
        self.info_card.language_text(index)

        self.presion_plot_card.language_text(index)

        self.opciones_card.language_text(index)
        self.opciones_plot_label.language_text(index)
        self.regiones_chip.language_text(index)
        self.presion_pico_local_chip.language_text(index)
        self.baricentros_chip.language_text(index)
        self.distribucion_chip.language_text(index)
        self.opciones_results_label.language_text(index)
        self.picos_button.language_text(index)
        self.means_button.language_text(index)
        self.areas_button.language_text(index)

        self.globales_card.language_text(index)
        self.presion_total_label.language_text(index)
        self.presion_left_label.language_text(index)
        self.presion_right_label.language_text(index)
        self.presion_antepie_label.language_text(index)
        self.presion_retropie_label.language_text(index)

        self.regiones_card.language_text(index)
        self.talon_interno_label.language_text(index)
        self.talon_externo_label.language_text(index)
        self.mediopie_interno_label.language_text(index)
        self.mediopie_externo_label.language_text(index)
        self.metatarsiano_1_label.language_text(index)
        self.metatarsiano_2_label.language_text(index)
        self.metatarsiano_3_label.language_text(index)
        self.metatarsiano_4_label.language_text(index)
        self.metatarsiano_5_label.language_text(index)
        self.dedo_1_label.language_text(index)
        self.dedo_2_label.language_text(index)
        self.dedo_3_5_label.language_text(index)

        self.settings.setValue('language', str(index))
        self.language_value = int(self.settings.value('language'))


    def on_tema_switch_clicked(self, state: bool) -> None:
        """ Theme switch control to change components stylesheet
        
        Parameters
        ----------
        state: bool
            State of theme switch control
        
        Returns
        -------
        None
        """
        if state: self.setStyleSheet('background-color: #E5E9F0; color: #000000')
        else: self.setStyleSheet('background-color: #3B4253; color: #E5E9F0')

        self.titulo_card.apply_styleSheet(state)
        self.idioma_menu.apply_styleSheet(state)
        self.tema_switch.set_state(state)
        self.tema_switch.apply_styleSheet(state)
        self.database_button.apply_styleSheet(state)
        self.manual_button.apply_styleSheet(state)
        self.about_button.apply_styleSheet(state)
        self.aboutQt_button.apply_styleSheet(state)

        self.paciente_card.apply_styleSheet(state)
        self.paciente_add_button.apply_styleSheet(state)
        self.paciente_edit_button.apply_styleSheet(state)
        self.paciente_del_button.apply_styleSheet(state)
        self.pacientes_menu.apply_styleSheet(state)

        self.analisis_card.apply_styleSheet(state)
        self.analisis_add_button.apply_styleSheet(state)
        self.analisis_del_button.apply_styleSheet(state)
        self.analisis_menu.apply_styleSheet(state)

        self.info_card.apply_styleSheet(state)
        self.apellido_value.apply_styleSheet(state)
        self.nombre_value.apply_styleSheet(state)
        self.id_label.apply_styleSheet(state)
        self.id_label.set_icon('id', state)
        self.id_value.apply_styleSheet(state)
        self.fecha_label.apply_styleSheet(state)
        self.fecha_label.set_icon('calendar', state)
        self.fecha_value.apply_styleSheet(state)
        self.sex_label.apply_styleSheet(state)
        self.sex_value.apply_styleSheet(state)
        
        if self.sex_value.text() == 'F': self.sex_label.set_icon('woman', state)
        elif self.sex_value.text() == 'M': self.sex_label.set_icon('man', state)

        self.peso_label.apply_styleSheet(state)
        self.peso_label.set_icon('weight', state)
        self.peso_value.apply_styleSheet(state)
        self.altura_label.apply_styleSheet(state)
        self.altura_label.set_icon('height', state)
        self.altura_value.apply_styleSheet(state)
        self.bmi_value.apply_styleSheet(state)

        self.presion_plot_card.apply_styleSheet(state)

    #     self.somatotipo_plot.apply_styleSheet(state)
    #     if self.lat_text_1:
    #         self.lat_text_1.remove()
    #         self.lat_text_2.remove()
    #         if state:
    #             self.lat_text_1 = self.somatotipo_plot.axes.text(self.data_lat_t_max, self.data_lat_max, f'{self.data_lat_max:.2f}', color='#000000')
    #             self.lat_text_2 = self.somatotipo_plot.axes.text(self.data_lat_t_min, self.data_lat_min, f'{self.data_lat_min:.2f}', color='#000000')
    #         else:
    #             self.lat_text_1 = self.somatotipo_plot.axes.text(self.data_lat_t_max, self.data_lat_max, f'{self.data_lat_max:.2f}', color='#E5E9F0')
    #             self.lat_text_2 = self.somatotipo_plot.axes.text(self.data_lat_t_min, self.data_lat_min, f'{self.data_lat_min:.2f}', color='#E5E9F0')
    #     self.somatotipo_plot.draw()

        self.opciones_card.apply_styleSheet(state)
        self.opciones_plot_label.apply_styleSheet(state)
        self.regiones_chip.apply_styleSheet(state)
        self.presion_pico_local_chip.apply_styleSheet(state)
        self.baricentros_chip.apply_styleSheet(state)
        self.distribucion_chip.apply_styleSheet(state)
        self.opciones_results_label.apply_styleSheet(state)
        self.picos_button.apply_styleSheet(state)
        self.means_button.apply_styleSheet(state)
        self.areas_button.apply_styleSheet(state)

        self.globales_card.apply_styleSheet(state)
        self.presion_total_label.apply_styleSheet(state)
        self.presion_total_value.apply_styleSheet(state)
        self.presion_total_percent.apply_styleSheet(state)
        self.presion_left_label.apply_styleSheet(state)
        self.presion_left_value.apply_styleSheet(state)
        self.presion_left_percent.apply_styleSheet(state)
        self.presion_right_label.apply_styleSheet(state)
        self.presion_right_value.apply_styleSheet(state)
        self.presion_right_percent.apply_styleSheet(state)
        self.presion_antepie_label.apply_styleSheet(state)
        self.presion_antepie_value.apply_styleSheet(state)
        self.presion_antepie_percent.apply_styleSheet(state)
        self.presion_retropie_label.apply_styleSheet(state)
        self.presion_retropie_value.apply_styleSheet(state)
        self.presion_retropie_percent.apply_styleSheet(state)
     
        self.regiones_card.apply_styleSheet(state)
        self.talon_interno_label.apply_styleSheet(state)
        self.talon_externo_label.apply_styleSheet(state)
        self.left_talon_interno_value.apply_styleSheet(state)
        self.right_talon_interno_value.apply_styleSheet(state)
        self.left_talon_externo_value.apply_styleSheet(state)
        self.right_talon_externo_value.apply_styleSheet(state)
        self.mediopie_interno_label.apply_styleSheet(state)
        self.mediopie_externo_label.apply_styleSheet(state)
        self.left_mediopie_interno_value.apply_styleSheet(state)
        self.right_mediopie_interno_value.apply_styleSheet(state)
        self.left_mediopie_externo_value.apply_styleSheet(state)
        self.right_mediopie_externo_value.apply_styleSheet(state)
        self.metatarsiano_1_label.apply_styleSheet(state)
        self.metatarsiano_2_label.apply_styleSheet(state)
        self.left_metatarsiano_1_value.apply_styleSheet(state)
        self.right_metatarsiano_1_value.apply_styleSheet(state)
        self.left_metatarsiano_2_value.apply_styleSheet(state)
        self.right_metatarsiano_2_value.apply_styleSheet(state)
        self.metatarsiano_3_label.apply_styleSheet(state)
        self.metatarsiano_4_label.apply_styleSheet(state)
        self.left_metatarsiano_3_value.apply_styleSheet(state)
        self.right_metatarsiano_3_value.apply_styleSheet(state)
        self.left_metatarsiano_4_value.apply_styleSheet(state)
        self.right_metatarsiano_4_value.apply_styleSheet(state)
        self.metatarsiano_5_label.apply_styleSheet(state)
        self.dedo_1_label.apply_styleSheet(state)
        self.left_metatarsiano_5_value.apply_styleSheet(state)
        self.right_metatarsiano_5_value.apply_styleSheet(state)
        self.left_dedo_1_value.apply_styleSheet(state)
        self.right_dedo_1_value.apply_styleSheet(state)
        self.dedo_2_label.apply_styleSheet(state)
        self.dedo_3_5_label.apply_styleSheet(state)
        self.left_dedo_2_value.apply_styleSheet(state)
        self.right_dedo_2_value.apply_styleSheet(state)
        self.left_dedo_3_5_value.apply_styleSheet(state)
        self.right_dedo_3_5_value.apply_styleSheet(state)

        self.left_talon_interno_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_talon_interno_value.setStyleSheet(f'QLabel#{self.left_talon_interno_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_talon_interno_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_talon_interno_value.setStyleSheet(f'QLabel#{self.right_talon_interno_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_talon_externo_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_talon_externo_value.setStyleSheet(f'QLabel#{self.left_talon_externo_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_talon_externo_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_talon_externo_value.setStyleSheet(f'QLabel#{self.right_talon_externo_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_mediopie_interno_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_mediopie_interno_value.setStyleSheet(f'QLabel#{self.left_mediopie_interno_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_mediopie_interno_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_mediopie_interno_value.setStyleSheet(f'QLabel#{self.right_mediopie_interno_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_mediopie_externo_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_mediopie_externo_value.setStyleSheet(f'QLabel#{self.left_mediopie_externo_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_mediopie_externo_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_mediopie_externo_value.setStyleSheet(f'QLabel#{self.right_mediopie_externo_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_metatarsiano_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_1_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_1_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_1_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_1_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_metatarsiano_2_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_2_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_2_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_2_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_2_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_2_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_metatarsiano_3_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_3_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_3_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_3_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_3_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_3_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_metatarsiano_4_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_4_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_4_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_4_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_4_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_4_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_metatarsiano_5_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_metatarsiano_5_value.setStyleSheet(f'QLabel#{self.left_metatarsiano_5_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_metatarsiano_5_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_metatarsiano_5_value.setStyleSheet(f'QLabel#{self.right_metatarsiano_5_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_dedo_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_dedo_1_value.setStyleSheet(f'QLabel#{self.left_dedo_1_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_dedo_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_dedo_1_value.setStyleSheet(f'QLabel#{self.right_dedo_1_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_dedo_2_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_dedo_2_value.setStyleSheet(f'QLabel#{self.left_dedo_2_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_dedo_2_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_dedo_2_value.setStyleSheet(f'QLabel#{self.right_dedo_2_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')
        self.left_dedo_3_5_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_dedo_3_5_value.setStyleSheet(f'QLabel#{self.left_dedo_3_5_value.name} {{'
            f'border: 2px solid #FF0000; border-radius: 16 }}')
        self.right_dedo_3_5_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_dedo_3_5_value.setStyleSheet(f'QLabel#{self.right_dedo_3_5_value.name} {{'
            f'border: 2px solid #0000FF; border-radius: 16 }}')

        self.settings.setValue('theme', f'{state}')
        self.theme_value = eval(self.settings.value('theme'))


    def on_database_button_clicked(self) -> None:
        """ Database button to configure the database """
        self.db_info = database.Database()
        self.db_info.exec()
        
        if self.db_info.database_data:
            self.patientes_list = backend.create_db('pacientes')
            self.estudios_list = backend.create_db('estudios')

            for data in self.patientes_list:
                self.pacientes_menu.addItem(str(data[4]))
            self.pacientes_menu.setCurrentIndex(-1)

            self.pacientes_menu.setEnabled(True)
            self.paciente_add_button.setEnabled(True)
            self.paciente_edit_button.setEnabled(True)
            self.paciente_del_button.setEnabled(True)

            if self.language_value == 0:
                QtWidgets.QMessageBox.information(self, 'Datos Guardados', 'Base de datos configurada')
            elif self.language_value == 1:
                QtWidgets.QMessageBox.information(self, 'Data Saved', 'Database configured')
        else:
            if self.language_value == 0:
                QtWidgets.QMessageBox.critical(self, 'Error de Datos', 'No se dio información de la base de datos')
            elif self.language_value == 1:
                QtWidgets.QMessageBox.critical(self, 'Data Error', 'No information on the database was given')


    def on_manual_button_clicked(self) -> None:
        """ Manual button to open manual window """
        return 0


    def on_about_button_clicked(self) -> None:
        """ About app button to open about app window dialog """
        self.about = backend.AboutApp()
        self.about.exec()


    def on_aboutQt_button_clicked(self) -> None:
        """ About Qt button to open about Qt window dialog """
        backend.about_qt_dialog(self, self.language_value)
        
    
    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        """ Resize event to control size and position of app components """
        width = self.geometry().width()
        height = self.geometry().height()

        self.titulo_card.resize(width - 16, 48)
        self.titulo_card.title.resize(width - 304, 32)
        self.idioma_menu.move(width - 312, 8)
        self.tema_switch.move(width - 232, 8)
        self.database_button.move(width - 176, 8)
        self.manual_button.move(width - 136, 8)
        self.about_button.move(width - 96, 8)
        self.aboutQt_button.move(width - 56, 8)

        self.presion_plot_card.setGeometry(196, 64, width - 636, int(height - 248))
        self.presion_plot_card.title.resize(width - 652, 32)
        # self.somatotipo_plot.setGeometry(8, 48, self.presion_plot_card.width()-16, self.presion_plot_card.height()-56)
       
        self.opciones_card.setGeometry(196, self.presion_plot_card.height()+72, width - 636, 168)
        self.globales_card.setGeometry(width - 432, 64, 424, 216)
        self.regiones_card.setGeometry(width - 432, 288, 424, 384)
        

        return super().resizeEvent(a0)

    # ------------------
    # Funciones Paciente
    # ------------------
    def on_paciente_add_button_clicked(self) -> None:
        """ Add patient button to the database """
        self.patient_window = patient.Patient()
        self.patient_window.exec()
        
        if self.patient_window.patient_data:
            if self.patient_window.patient_data['sex'] == 'F':
                self.sex_label.set_icon('woman', self.theme_value)
            elif self.patient_window.patient_data['sex'] == 'M':
                self.sex_label.set_icon('man', self.theme_value)

            self.apellido_value.setText(self.patient_window.patient_data['last_name'])
            self.nombre_value.setText(self.patient_window.patient_data['first_name'])
            self.id_value.setText(f'{self.patient_window.patient_data["id_type"]} {self.patient_window.patient_data["id"]}')
            self.fecha_value.setText(self.patient_window.patient_data['birth_date'])
            self.sex_value.setText(self.patient_window.patient_data['sex'])
            self.peso_value.setText(f'{self.patient_window.patient_data["weight"]} {self.patient_window.patient_data["weight_unit"]}')
            self.altura_value.setText(f'{self.patient_window.patient_data["height"]} {self.patient_window.patient_data["height_unit"]}')
            self.bmi_value.setText(self.patient_window.patient_data['bmi'])

    #         # -------------
    #         # Base de datos
    #         # -------------
    #         self.patientes_list = backend.add_db('pacientes', self.patient_window.patient_data)
            
    #         self.pacientes_menu.clear()
    #         for data in self.patientes_list:
    #             self.pacientes_menu.addItem(str(data[4]))
    #         self.pacientes_menu.setCurrentIndex(len(self.patientes_list)-1)

    #         self.analisis_add_button.setEnabled(True)
    #         self.analisis_del_button.setEnabled(True)
    #         self.analisis_menu.setEnabled(True)

    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.information(self, 'Datos Guardados', 'Paciente agregado a la base de datos')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.information(self, 'Data Saved', 'Patient added to database')
    #     else:
    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.critical(self, 'Error de Datos', 'No se dio información de un paciente nuevo')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.critical(self, 'Data Error', 'No information on a new patient was given')


    # def on_paciente_edit_button_clicked(self) -> None:
    #     """ Edit patient button in the database """
    #     patient_id = self.pacientes_menu.currentText()

    #     if patient_id != '':
    #         patient_data = backend.get_db('pacientes', patient_id)

    #         id_db = patient_data[0][0]
    #         self.patient_window = patient.Patient()
    #         self.patient_window.apellido_text.text_field.setText(patient_data[0][1])
    #         self.patient_window.nombre_text.text_field.setText(patient_data[0][2])
    #         if patient_data[0][3] == 'CC':
    #             self.patient_window.cc_button.set_state(True)
    #         elif patient_data[0][3] == 'TI':
    #             self.patient_window.ti_button.set_state(True)
    #         self.patient_window.id_text.text_field.setText(str(patient_data[0][4]))
    #         self.patient_window.fecha_date.text_field.setDate(QtCore.QDate.fromString(patient_data[0][5], 'dd/MM/yyyy'))
    #         if patient_data[0][6] == 'F':
    #             self.patient_window.f_button.set_state(True)
    #         elif patient_data[0][6] == 'M':
    #             self.patient_window.m_button.set_state(True)
    #         self.patient_window.peso_text.text_field.setText(str(patient_data[0][7]))
    #         if patient_data[0][8] == 'Kg':
    #             self.patient_window.kg_button.set_state(True)
    #         elif patient_data[0][8] == 'Lb':
    #             self.patient_window.lb_button.set_state(True)
    #         self.patient_window.altura_text.text_field.setText(str(patient_data[0][9]))
    #         if patient_data[0][10] == 'm':
    #             self.patient_window.mt_button.set_state(True)
    #         elif patient_data[0][10] == 'ft - in':
    #             self.patient_window.fi_button.set_state(True)
    #         self.patient_window.bmi_value_label.setText(str(patient_data[0][11]))

    #         self.patient_window.exec()

    #         if self.patient_window.patient_data:
    #             self.patientes_list = backend.edit_db('pacientes', id_db, self.patient_window.patient_data)

    #             self.pacientes_menu.clear()
    #             for data in self.patientes_list:
    #                 self.pacientes_menu.addItem(str(data[4]))
    #             self.pacientes_menu.setCurrentIndex(-1)

    #             self.analisis_add_button.setEnabled(False)
    #             self.analisis_del_button.setEnabled(False)
    #             self.analisis_menu.setEnabled(False)

    #             self.apellido_value.setText('')
    #             self.nombre_value.setText('')
    #             self.id_value.setText('')
    #             self.fecha_value.setText('')
    #             self.sex_value.setText('')
    #             self.sex_label.set_icon('', self.theme_value)
    #             self.peso_value.setText('')
    #             self.altura_value.setText('')
    #             self.bmi_value.setText('')

    #             if self.language_value == 0:
    #                 QtWidgets.QMessageBox.information(self, 'Datos Guardados', 'Paciente editado en la base de datos')
    #             elif self.language_value == 1:
    #                 QtWidgets.QMessageBox.information(self, 'Data Saved', 'Patient edited in database')
    #         else:
    #             if self.language_value == 0:
    #                 QtWidgets.QMessageBox.critical(self, 'Error de Datos', 'No se dio información del paciente')
    #             elif self.language_value == 1:
    #                 QtWidgets.QMessageBox.critical(self, 'Data Error', 'No information on a patient was given')
    #     else:
    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.critical(self, 'Error de Paciente', 'No se seleccionó un paciente')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.critical(self, 'Patient Error', 'No patient selected')


    # def on_paciente_del_button_clicked(self) -> None:
    #     """ Delete patient button from the database """
    #     patient_id = self.pacientes_menu.currentText()

    #     if patient_id != '':
    #         self.patientes_list = backend.delete_db('pacientes', patient_id)

    #         self.pacientes_menu.clear()
    #         for data in self.patientes_list:
    #             self.pacientes_menu.addItem(str(data[4]))
    #         self.pacientes_menu.setCurrentIndex(-1)

    #         self.analisis_add_button.setEnabled(False)
    #         self.analisis_del_button.setEnabled(False)
    #         self.analisis_menu.setEnabled(False)

    #         self.apellido_value.setText('')
    #         self.nombre_value.setText('')
    #         self.id_value.setText('')
    #         self.fecha_value.setText('')
    #         self.sex_value.setText('')
    #         self.sex_label.set_icon('', self.theme_value)
    #         self.peso_value.setText('')
    #         self.altura_value.setText('')
    #         self.bmi_value.setText('')

    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.information(self, 'Datos Guardados', 'Paciente eliminado de la base de datos')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.information(self, 'Data Saved', 'Patient deleted from database')
    #     else:
    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.critical(self, 'Error de Paciente', 'No se seleccionó un paciente')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.critical(self, 'Patient Error', 'No patient selected')


    # def on_pacientes_menu_textActivated(self, current_pacient: str) -> None:
    #     """ Change active patient and present previously saved studies and information
        
    #     Parameters
    #     ----------
    #     current_pacient: str
    #         Current pacient text
        
    #     Returns
    #     -------
    #     None
    #     """
    #     patient_data = backend.get_db('pacientes', current_pacient)

    #     if patient_data[0][6] == 'F':
    #         self.sex_label.set_icon('woman', self.theme_value)
    #     elif patient_data[0][6] == 'M':
    #         self.sex_label.set_icon('man', self.theme_value)

    #     self.apellido_value.setText(patient_data[0][1])
    #     self.nombre_value.setText(patient_data[0][2])
    #     self.id_value.setText(f'{patient_data[0][3]} {patient_data[0][4]}')
    #     self.fecha_value.setText(patient_data[0][5])
    #     self.sex_value.setText(patient_data[0][6])
    #     self.peso_value.setText(f'{patient_data[0][7]} {patient_data[0][8]}')
    #     self.altura_value.setText(f'{patient_data[0][9]} {patient_data[0][10]}')
    #     self.bmi_value.setText(str(patient_data[0][11]))

    #     self.analisis_add_button.setEnabled(True)
    #     self.analisis_del_button.setEnabled(True)
    #     self.analisis_menu.setEnabled(True)

    #     self.estudios_list = backend.get_db('estudios', current_pacient)
    #     self.analisis_menu.clear()
    #     for data in self.estudios_list:
    #         self.analisis_menu.addItem(data[2])
    #     self.analisis_menu.setCurrentIndex(-1)

    #     self.lateral_plot.axes.cla()
    #     self.lateral_plot.draw()
    #     self.antePost_plot.axes.cla()
    #     self.antePost_plot.draw()
    #     self.elipse_plot.axes.cla()
    #     self.elipse_plot.draw()
    #     self.hull_plot.axes.cla()
    #     self.hull_plot.draw()
    #     self.pca_plot.axes.cla()
    #     self.pca_plot.draw()

    #     self.lat_rango_value.setText('')
    #     self.lat_vel_value.setText('')
    #     self.lat_rms_value.setText('')
    #     self.ap_rango_value.setText('')
    #     self.ap_vel_value.setText('')
    #     self.ap_rms_value.setText('')
    #     self.cop_vel_value.setText('')
    #     self.distancia_value.setText('')
    #     self.frecuencia_value.setText('')
    #     self.elipse_value.setText('')
    #     self.hull_value.setText('')
    #     self.pca_value.setText('')

    
    # -----------------
    # Funciones Estudio
    # -----------------
    # def on_analisis_add_button_clicked(self) -> None:
    #     """ Add analysis button to the database """
    #     selected_file = QtWidgets.QFileDialog.getOpenFileName(None,
    #             'Seleccione el archivo de datos', 'D:/Data/Asdhar Alonso/',
    #             'Archivos de Datos (*.csv *.txt *.emt)')[0]

    #     if selected_file:
    #         df = pd.read_csv(selected_file, sep='\t', skiprows=43, encoding='ISO-8859-1')

    #         results = backend.analisis(df)
            
    #         # ----------------
    #         # Gráficas Señales
    #         # ----------------
    #         data_lat = results['data_x']
    #         data_ap = results['data_y']
    #         data_t = results['data_t']

    #         self.data_lat_max = results['lat_max']
    #         self.data_lat_t_max = results['lat_t_max']
    #         self.data_lat_min = results['lat_min']
    #         self.data_lat_t_min = results['lat_t_min']

    #         self.lateral_plot.axes.cla()
    #         self.lateral_plot.fig.subplots_adjust(left=0.05, bottom=0.15, right=1, top=0.95, wspace=0, hspace=0)
    #         self.lateral_plot.axes.plot(data_t, data_lat, '#42A4F5')
    #         self.lateral_plot.axes.plot(self.data_lat_t_max, self.data_lat_max, marker="o", markersize=3, markeredgecolor='#FF2D55', markerfacecolor='#FF2D55')
    #         self.lateral_plot.axes.plot(self.data_lat_t_min, self.data_lat_min, marker="o", markersize=3, markeredgecolor='#FF2D55', markerfacecolor='#FF2D55')
    #         if self.theme_value:
    #             self.lat_text_1 = self.lateral_plot.axes.text(self.data_lat_t_max, self.data_lat_max, f'{self.data_lat_max:.2f}', color='#000000')
    #             self.lat_text_2 = self.lateral_plot.axes.text(self.data_lat_t_min, self.data_lat_min, f'{self.data_lat_min:.2f}', color='#000000')
    #         else:
    #             self.lat_text_1 = self.lateral_plot.axes.text(self.data_lat_t_max, self.data_lat_max, f'{self.data_lat_max:.2f}', color='#E5E9F0')
    #             self.lat_text_2 = self.lateral_plot.axes.text(self.data_lat_t_min, self.data_lat_min, f'{self.data_lat_min:.2f}', color='#E5E9F0')
    #         self.lateral_plot.draw()

    #         self.data_ap_max = results['ap_max']
    #         self.data_ap_t_max = results['ap_t_max']
    #         self.data_ap_min = results['ap_min']
    #         self.data_ap_t_min = results['ap_t_min']

    #         self.antePost_plot.axes.cla()
    #         self.antePost_plot.fig.subplots_adjust(left=0.05, bottom=0.15, right=1, top=0.95, wspace=0, hspace=0)
    #         self.antePost_plot.axes.plot(data_t, data_ap, '#42A4F5')
    #         self.antePost_plot.axes.plot(self.data_ap_t_max, self.data_ap_max, marker="o", markersize=3, markeredgecolor='#FF2D55', markerfacecolor='#FF2D55')
    #         self.antePost_plot.axes.plot(self.data_ap_t_min, self.data_ap_min, marker="o", markersize=3, markeredgecolor='#FF2D55', markerfacecolor='#FF2D55')
    #         if self.theme_value:
    #             self.ap_text_1 = self.antePost_plot.axes.text(self.data_ap_t_max, self.data_ap_max, f'{self.data_ap_max:.2f}', color='#000000')
    #             self.ap_text_2 = self.antePost_plot.axes.text(self.data_ap_t_min, self.data_ap_min, f'{self.data_ap_min:.2f}', color='#000000')
    #         else:
    #             self.ap_text_1 = self.antePost_plot.axes.text(self.data_ap_t_max, self.data_ap_max, f'{self.data_ap_max:.2f}', color='#E5E9F0')
    #             self.ap_text_2 = self.antePost_plot.axes.text(self.data_ap_t_min, self.data_ap_min, f'{self.data_ap_min:.2f}', color='#E5E9F0')
    #         self.antePost_plot.draw()

    #         # --------------
    #         # Gráficas Áreas
    #         # --------------
    #         data_elipse = backend.ellipseStandard(df)
    #         self.elipse_plot.axes.cla()
    #         self.elipse_plot.fig.subplots_adjust(left=0.1, bottom=0.1, right=1, top=0.95, wspace=0, hspace=0)
    #         self.elipse_plot.axes.scatter(data_lat, data_ap, marker='.', color='#42A4F5')
    #         self.elipse_plot.axes.plot(data_elipse['x'], data_elipse['y'], '#FF2D55')
    #         self.elipse_plot.axes.axis('equal')
    #         self.elipse_plot.draw()

    #         data_convex = backend.convexHull(df)
    #         self.hull_plot.axes.cla()
    #         self.hull_plot.fig.subplots_adjust(left=0.1, bottom=0.1, right=1, top=0.95, wspace=0, hspace=0)
    #         self.hull_plot.axes.scatter(data_lat, data_ap, marker='.', color='#42A4F5')
    #         self.hull_plot.axes.fill(data_convex['x'], data_convex['y'], edgecolor='#FF2D55', fill=False, linewidth=2)
    #         self.hull_plot.axes.axis('equal')
    #         self.hull_plot.draw()

    #         data_pca = backend.ellipsePCA(df)
    #         self.pca_plot.axes.cla()
    #         self.pca_plot.fig.subplots_adjust(left=0.1, bottom=0.1, right=1, top=0.95, wspace=0, hspace=0)
    #         self.pca_plot.axes.scatter(data_lat, data_ap, marker='.', color='#42A4F5')
    #         self.pca_plot.axes.plot(data_pca['x'], data_pca['y'], '#FF2D55')
    #         self.pca_plot.axes.axis('equal')
    #         self.pca_plot.draw()

    #         # --------------------------
    #         # Presentación de resultados
    #         # --------------------------
    #         self.lat_rango_value.setText(f'{results["lat_rango"]:.2f}')
    #         self.lat_vel_value.setText(f'{results["lat_vel"]:.2f}')
    #         self.lat_rms_value.setText(f'{results["lat_rms"]:.2f}')

    #         self.ap_rango_value.setText(f'{results["ap_rango"]:.2f}')
    #         self.ap_vel_value.setText(f'{results["ap_vel"]:.2f}')
    #         self.ap_rms_value.setText(f'{results["ap_rms"]:.2f}')

    #         self.cop_vel_value.setText(f'{results["centro_vel"]:.2f}')
    #         self.distancia_value.setText(f'{results["centro_dist"]:.2f}')
    #         self.frecuencia_value.setText(f'{results["centro_frec"]:.2f}')

    #         self.elipse_value.setText(f'{data_elipse["area"]:.2f}')
    #         self.hull_value.setText(f'{data_convex["area"]:.2f}')
    #         self.pca_value.setText(f'{data_pca["area"]:.2f}')

    #         # -------------
    #         # Base de datos
    #         # -------------
    #         study_data = {
    #             'id_number': self.pacientes_menu.currentText(),
    #             'file_name': Path(selected_file).name,
    #             'file_path': selected_file
    #             }
    #         self.estudios_list = backend.add_db('estudios', study_data)
            
    #         self.analisis_menu.clear()
    #         for data in self.estudios_list:
    #             self.analisis_menu.addItem(data[2])
    #         self.analisis_menu.setCurrentIndex(len(self.patientes_list)-1)

    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.information(self, 'Datos Guardados', 'Estudio agregado a la base de datos')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.information(self, 'Data Saved', 'Study added to database')
    #     else:
    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.critical(self, 'Error de Datos', 'No se seleccióno un archivo para el estudio')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.critical(self, 'Data Error', 'No file for a study was given')


    # def on_analisis_del_button_clicked(self) -> None:
    #     """ Delete analysis button from the database """
    #     current_study = self.analisis_menu.currentText()

    #     if current_study != '':
    #         self.estudios_list = backend.delete_db('estudios', current_study)
            
    #         self.analisis_menu.clear()
    #         for data in self.estudios_list:
    #             self.analisis_menu.addItem(data[2])
    #         self.analisis_menu.setCurrentIndex(-1)

    #         self.lateral_plot.axes.cla()
    #         self.lateral_plot.draw()
    #         self.antePost_plot.axes.cla()
    #         self.antePost_plot.draw()
    #         self.elipse_plot.axes.cla()
    #         self.elipse_plot.draw()
    #         self.hull_plot.axes.cla()
    #         self.hull_plot.draw()
    #         self.pca_plot.axes.cla()
    #         self.pca_plot.draw()

    #         self.lat_rango_value.setText('')
    #         self.lat_vel_value.setText('')
    #         self.lat_rms_value.setText('')
    #         self.ap_rango_value.setText('')
    #         self.ap_vel_value.setText('')
    #         self.ap_rms_value.setText('')
    #         self.cop_vel_value.setText('')
    #         self.distancia_value.setText('')
    #         self.frecuencia_value.setText('')
    #         self.elipse_value.setText('')
    #         self.hull_value.setText('')
    #         self.pca_value.setText('')

    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.information(self, 'Datos Guardados', 'Análisis eliminado de la base de datos')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.information(self, 'Data Saved', 'Analysis deleted from database')
    #     else:
    #         if self.language_value == 0:
    #             QtWidgets.QMessageBox.critical(self, 'Error de Análisis', 'No se seleccionó un análisis')
    #         elif self.language_value == 1:
    #             QtWidgets.QMessageBox.critical(self, 'Analysis Error', 'No analysis selected')


    # def on_analisis_menu_textActivated(self, current_study: str):
    #     """ Change analysis and present results
        
    #     Parameters
    #     ----------
    #     current_study: str
    #         Current study text
        
    #     Returns
    #     -------
    #     None
    #     """
    #     analisis_data = backend.get_db('estudios', self.pacientes_menu.currentText())
    #     study_path = [item for item in analisis_data if item[2] == current_study][0][3]

    #     df = pd.read_csv(study_path, sep='\t', skiprows=43, encoding='ISO-8859-1')

    #     results = backend.analisis(df)
        
    #     # ----------------
    #     # Gráficas Señales
    #     # ----------------
    #     data_lat = results['data_x']
    #     data_ap = results['data_y']
    #     data_t = results['data_t']

    #     self.data_lat_max = results['lat_max']
    #     self.data_lat_t_max = results['lat_t_max']
    #     self.data_lat_min = results['lat_min']
    #     self.data_lat_t_min = results['lat_t_min']

    #     self.lateral_plot.axes.cla()
    #     self.lateral_plot.fig.subplots_adjust(left=0.05, bottom=0.1, right=1, top=0.95, wspace=0, hspace=0)
    #     self.lateral_plot.axes.plot(data_t, data_lat, '#42A4F5')
    #     self.lateral_plot.axes.plot(self.data_lat_t_max, self.data_lat_max, marker="o", markersize=3, markeredgecolor='#FF2D55', markerfacecolor='#FF2D55')
    #     self.lateral_plot.axes.plot(self.data_lat_t_min, self.data_lat_min, marker="o", markersize=3, markeredgecolor='#FF2D55', markerfacecolor='#FF2D55')
    #     if self.theme_value:
    #         self.lat_text_1 = self.lateral_plot.axes.text(self.data_lat_t_max, self.data_lat_max, f'{self.data_lat_max:.2f}', color='#000000')
    #         self.lat_text_2 = self.lateral_plot.axes.text(self.data_lat_t_min, self.data_lat_min, f'{self.data_lat_min:.2f}', color='#000000')
    #     else:
    #         self.lat_text_1 = self.lateral_plot.axes.text(self.data_lat_t_max, self.data_lat_max, f'{self.data_lat_max:.2f}', color='#E5E9F0')
    #         self.lat_text_2 = self.lateral_plot.axes.text(self.data_lat_t_min, self.data_lat_min, f'{self.data_lat_min:.2f}', color='#E5E9F0')
    #     self.lateral_plot.draw()

    #     self.data_ap_max = results['ap_max']
    #     self.data_ap_t_max = results['ap_t_max']
    #     self.data_ap_min = results['ap_min']
    #     self.data_ap_t_min = results['ap_t_min']

    #     # --------------------------
    #     # Presentación de resultados
    #     # --------------------------
    #     self.lat_rango_value.setText(f'{results["lat_rango"]:.2f}')
    #     self.lat_vel_value.setText(f'{results["lat_vel"]:.2f}')
    #     self.lat_rms_value.setText(f'{results["lat_rms"]:.2f}')

    #     self.ap_rango_value.setText(f'{results["ap_rango"]:.2f}')
    #     self.ap_vel_value.setText(f'{results["ap_vel"]:.2f}')
    #     self.ap_rms_value.setText(f'{results["ap_rms"]:.2f}')

    #     self.cop_vel_value.setText(f'{results["centro_vel"]:.2f}')
    #     self.distancia_value.setText(f'{results["centro_dist"]:.2f}')
    #     self.frecuencia_value.setText(f'{results["centro_frec"]:.2f}')

    #     self.elipse_value.setText(f'{data_elipse["area"]:.2f}')
    #     self.hull_value.setText(f'{data_convex["area"]:.2f}')
    #     self.pca_value.setText(f'{data_pca["area"]:.2f}')


if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())
