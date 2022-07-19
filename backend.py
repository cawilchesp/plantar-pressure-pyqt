"""
Backend

This file contains supplementary methods and classes applied to the frontend.

1. Class MPLCanvas: configuration of the plot canvas
2. Analysis methods: methods to process and analyze anthropometric measurements
3. Database methods: methods of the database operations
4. About class and method: Dialogs of information about me and Qt

"""

from PyQt6 import QtWidgets
from PyQt6.QtCore import QSettings

import sys
import numpy as np
import pandas as pd
import psycopg2
import cv2

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import material3_components as mt3

light = {
    'surface': '#B2B2B2',
    'on_surface': '#000000'
}

dark = {
    'surface': '#2E3441',
    'on_surface': '#E5E9F0'
}

class MPLCanvas(FigureCanvasQTAgg):
    def __init__(self, parent, theme: bool) -> None:
        """ Canvas settings for plotting signals """
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)

        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)

        self.apply_styleSheet(theme)

    def apply_styleSheet(self, theme):
        self.fig.subplots_adjust(left=0.05, bottom=0.15, right=1, top=0.95, wspace=0, hspace=0)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.axes.spines['left'].set_visible(False)
        if theme:
            self.fig.set_facecolor(f'{light["surface"]}')
            self.axes.set_facecolor(f'{light["surface"]}')
            self.axes.xaxis.label.set_color(f'{light["on_surface"]}')
            self.axes.yaxis.label.set_color(f'{light["on_surface"]}')
            self.axes.tick_params(axis='both', colors=f'{light["on_surface"]}', labelsize=8)
        else:
            self.fig.set_facecolor(f'{dark["surface"]}')
            self.axes.set_facecolor(f'{dark["surface"]}')
            self.axes.xaxis.label.set_color(f'{dark["on_surface"]}')
            self.axes.yaxis.label.set_color(f'{dark["on_surface"]}')
            self.axes.tick_params(axis='both', colors=f'{dark["on_surface"]}', labelsize=8)

# -----------------------
# Extracción de la Imagen
# -----------------------
def extract(left_image_file: str, right_image_file: str) -> dict:
    """ Extraction of pressure image from pressure data files 
    
    Parameters
    ----------
    left_image_file: str
        Input data file path of left foot

    right_image_file: str
        Input data file path of right foot

    Returns
    -------
    signals: dict
        Lateral and antero-posterior signal data by feet
    """
    left_mdata = {}
    with open(left_image_file) as f:
        lines = f.readlines()
        left_row = int(lines[19].split('=')[1]) - 1
        left_col = int(lines[20].split('=')[1]) - 1
        left_height = int(lines[21].split('=')[1])
        left_width = int(lines[22].split('=')[1])
    left_mdata['row'] = left_row
    left_mdata['col'] = left_col
    left_mdata['height'] = left_height
    left_mdata['width'] = left_width

    right_mdata = {}
    with open(right_image_file) as f:
        lines = f.readlines()
        right_row = int(lines[19].split('=')[1]) - 1
        right_col = int(lines[20].split('=')[1]) - 1
        right_height = int(lines[21].split('=')[1])
        right_width = int(lines[22].split('=')[1])
    right_mdata['row'] = right_row
    right_mdata['col'] = right_col
    right_mdata['height'] = right_height
    right_mdata['width'] = right_width

    left_df = pd.read_csv(left_image_file, sep='\t', skiprows=27, header=None, encoding='ISO-8859-1')
    right_df = pd.read_csv(right_image_file, sep='\t', skiprows=27, header=None, encoding='ISO-8859-1')

    left_df = np.array(left_df)
    left_df = np.nan_to_num(left_df,False,-1.0)

    right_df = np.array(right_df)
    right_df = np.nan_to_num(right_df,False,-1.0)

    pressure = np.zeros((48,48)) - 10
    pressure[left_row:left_row+left_height , left_col:left_col+left_width+1] = left_df * 10
    pressure[right_row:right_row+right_height , right_col:right_col+right_width+1] = right_df * 10
    

    results = analisis(left_df, right_df, pressure, left_mdata, right_mdata)

    # # OCR
    # image_left_limits = image.copy()
    # image_left_limits = image_left_limits[ 144:315 , 108:513]
    # left_ap_limits,left_lat_limits = image_ocr(image_left_limits)

  
    # signals = {
    #     'left_lateral_signal': left_lateral_signal,
    #     'left_lateral_time': left_lateral_time,
    #     'center_lateral_signal': center_lateral_signal,
    #     'center_lateral_time': center_lateral_time,
    #     }

    return pressure, results


# ---------------------------
# Funciones Análisis de Datos
# ---------------------------
def center_pressure(image):
    image[image<0] = 0.0
    i  = np.nonzero(image)
    res = np.vstack([i, image[i]])
    
    pressure_y = res[0] - 0.5
    pressure_x = res[1] - 0.5
    pressure_values = res[2]

    den = np.sum(pressure_values)
    cop_x = sum(pressure_values * pressure_x) / den
    cop_y = sum(pressure_values * pressure_y) / den

    return (cop_x, cop_y)


def analisis(left_df: np.array, right_df: np.array, pressure: np.array, left_mdata: dict, right_mdata: dict) -> dict:
    """ Analysis of anthropometric measurements

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe converted from balance signal data from file
    
    Returns
    -------
    results: dict
        Results of dataframe analysis of lateral, antero-posterior, and
        center of pressure oscillations
        data_x: pd.DataFrame
            Lateral signal
        data_y: pd.DataFrame
            Antero-posterior signal
        data_t: 
            Time signal
        lat_max: float
            Lateral signal maximum value
        lat_t_max: float
            Lateral signal correspondent time value for maximum value
        lat_min: float
            Lateral signal minimum value
        lat_t_min: float
            Lateral signal correspondent time value for minimum value
        
    """
    results = {}

    left_cop = center_pressure(left_df)
    right_cop = center_pressure(right_df)
    global_cop = center_pressure(pressure)

    results['left_cop'] = (left_cop[0] + left_mdata['col'] , left_cop[1] + left_mdata['row'])
    results['right_cop'] = (right_cop[0] + right_mdata['col'] , right_cop[1] + right_mdata['row'])
    results['global_cop'] = (global_cop[0] , global_cop[1])

    Q1 = pressure[ 0:int(global_cop[1])  , 0:int(global_cop[0]) ]
    Q2 = pressure[ int(global_cop[1]):47 , 0:int(global_cop[0]) ]
    Q3 = pressure[ 0:int(global_cop[1])  , int(global_cop[0]):47 ]
    Q4 = pressure[ int(global_cop[1]):47 , int(global_cop[0]):47 ]
    
    total_pressure = np.sum(pressure)
    pressure_Q1 = np.sum(Q1)
    pressure_Q2 = np.sum(Q2)
    pressure_Q3 = np.sum(Q3)
    pressure_Q4 = np.sum(Q4)

    results['total_pressure'] = total_pressure
    results['pressure_Q1'] = pressure_Q1
    results['pressure_Q2'] = pressure_Q2
    results['pressure_Q3'] = pressure_Q3
    results['pressure_Q4'] = pressure_Q4

    results['left_pressure'] = pressure_Q1 + pressure_Q2
    results['left_pressure_perc'] = (pressure_Q1 + pressure_Q2) * 100 / total_pressure
    results['right_pressure'] = pressure_Q3 + pressure_Q4
    results['right_pressure_perc'] = (pressure_Q3 + pressure_Q4) * 100 / total_pressure
    results['forefoot_pressure'] = pressure_Q1 + pressure_Q3
    results['forefoot_pressure_perc'] = (pressure_Q1 + pressure_Q3) * 100 / total_pressure
    results['rearfoot_pressure'] = pressure_Q2 + pressure_Q4
    results['rearfoot_pressure_perc'] = (pressure_Q2 + pressure_Q4) * 100 / total_pressure
    
    
    
    




    left_max = left_df.max().max()
    left_peak_pos = np.unravel_index(left_df.argmax(), left_df.shape)
    
    results['left_max'] = left_max
    results['left_peak_pos'] = (left_peak_pos[0] + left_mdata['row'], left_peak_pos[1] + left_mdata['col'])
    

    

    return results


# -----------------------
# Funciones Base de Datos
# -----------------------
def create_db(db_table: str) -> list:
    """ Creates database tables if they don't exist and returns table data
    
    Parameters
    ----------
    db_table: str
        Database table name
    
    Returns
    -------
    table_data: list
        Data of table if exists (empty if table don't exist)
    """
    settings = QSettings(f'{sys.path[0]}/settings.ini', QSettings.Format.IniFormat)
    db_host = settings.value('db_host')
    db_port = settings.value('db_port')
    db_name = settings.value('db_name')
    db_user = settings.value('db_user')
    db_password = settings.value('db_password')
    try:
        connection = psycopg2.connect(user=db_user, 
                                  password=db_password, 
                                  host=db_host, 
                                  port=db_port, 
                                  database=db_name)
    except psycopg2.OperationalError as err:
        return err

    cursor = connection.cursor()

    if db_table == 'pacientes':
        cursor.execute("""CREATE TABLE IF NOT EXISTS pacientes (
                        id serial PRIMARY KEY,
                        last_name VARCHAR(128) NOT NULL,
                        first_name VARCHAR(128) NOT NULL,
                        id_type CHAR(2) NOT NULL,
                        id_number BIGINT UNIQUE NOT NULL,
                        birth_date VARCHAR(128) NOT NULL,
                        sex CHAR(1) NOT NULL,
                        weight NUMERIC(5,2) NOT NULL,
                        weight_unit CHAR(2) NOT NULL,
                        height NUMERIC(3,2) NOT NULL,
                        height_unit VARCHAR(7) NOT NULL,
                        bmi NUMERIC(4,2) NOT NULL
                        )""")
    elif db_table == 'estudios':
        cursor.execute("""CREATE TABLE IF NOT EXISTS estudios (
                        id serial PRIMARY KEY,
                        id_number BIGINT NOT NULL,
                        triceps_endo SMALLINT NOT NULL,
                        subescapular_endo SMALLINT NOT NULL,
                        supraespinal_endo SMALLINT NOT NULL,
                        calf_endo SMALLINT NOT NULL,
                        altura_meso NUMERIC(3,2) NOT NULL,
                        humero_meso NUMERIC(4,2) NOT NULL,
                        femur_meso NUMERIC(4,2) NOT NULL,
                        biceps_meso NUMERIC(4,2) NOT NULL,
                        tricipital_meso NUMERIC(4,2) NOT NULL,
                        calf_perimetro_meso NUMERIC(4,2) NOT NULL,
                        calf_pliegue_meso NUMERIC(4,2) NOT NULL,
                        peso_ecto NUMERIC(5,2) NOT NULL
                        )""")

    connection.commit()

    table_data = None
    if db_table == 'pacientes':
        cursor.execute('SELECT * FROM pacientes ORDER BY id ASC')
        table_data = cursor.fetchall()
    
    connection.close()

    return table_data


def add_db(db_table: str, data: dict) -> list:
    """ Adds data to database table and returns table data updated
    
    Parameters
    ----------
    db_table: str
        Database table name
    data: dict
        Data from patient or study file
    
    Returns
    -------
    table_data: list
        Data of table updated
    """
    if db_table == 'pacientes':
        last_name_value = data['last_name']
        first_name_value = data['first_name']
        id_type_value = data['id_type']
        id_value = data['id']
        birth_date_value = data['birth_date']
        sex_value = data['sex']
        weight_value = data['weight']
        weight_unit = data['weight_unit']
        height_value = data['height']
        height_unit = data['height_unit']
        bmi_value = data['bmi']
    # elif db_table == 'estudios':
    #     id_value = data['id_number']
    #     file_name_value = data['file_name']
    #     file_path_value = data['file_path']

    settings = QSettings(f'{sys.path[0]}/settings.ini', QSettings.Format.IniFormat)
    db_host = settings.value('db_host')
    db_port = settings.value('db_port')
    db_name = settings.value('db_name')
    db_user = settings.value('db_user')
    db_password = settings.value('db_password')
    connection = psycopg2.connect(user=db_user, 
                                  password=db_password, 
                                  host=db_host, 
                                  port=db_port, 
                                  database=db_name)
    cursor = connection.cursor()

    insert_query = None
    if db_table == 'pacientes':
        insert_query = f"""INSERT INTO pacientes (last_name, first_name, id_type, id_number, birth_date, sex, weight, weight_unit, height, height_unit, bmi) 
                    VALUES ('{last_name_value}', '{first_name_value}', '{id_type_value}', '{id_value}', '{birth_date_value}', '{sex_value}', '{weight_value}', '{weight_unit}', '{height_value}', '{height_unit}', '{bmi_value}')"""
    # elif db_table == 'estudios':
    #     insert_query = f"""INSERT INTO estudios (id_number, file_name, file_path) 
    #                 VALUES ('{id_value}', '{file_name_value}', '{file_path_value}')"""

    cursor.execute(insert_query)
    connection.commit()

    table_data = None
    if db_table == 'pacientes':
        cursor.execute('SELECT * FROM pacientes ORDER BY id ASC')
        table_data = cursor.fetchall()
    # elif db_table == 'estudios':
    #     cursor.execute(f"SELECT * FROM estudios WHERE id_number='{id_value}' ORDER BY id ASC")
    #     table_data = cursor.fetchall()
    
    connection.close()

    return table_data


def get_db(db_table: str, data_id: str) -> list:
    """ Get data from database table
    
    Parameters
    ----------
    db_table: str
        Database table name
    data_id: str
        Patient id number
    
    Returns
    -------
    table_data: list
        Data of table
    """
    settings = QSettings(f'{sys.path[0]}/settings.ini', QSettings.Format.IniFormat)
    db_host = settings.value('db_host')
    db_port = settings.value('db_port')
    db_name = settings.value('db_name')
    db_user = settings.value('db_user')
    db_password = settings.value('db_password')
    connection = psycopg2.connect(user=db_user, 
                                  password=db_password, 
                                  host=db_host, 
                                  port=db_port, 
                                  database=db_name)
    cursor = connection.cursor()

    table_data = None
    if db_table == 'pacientes':
        cursor.execute(f"SELECT * FROM pacientes WHERE id_number='{data_id}'")
    elif db_table == 'estudios':
        cursor.execute(f"SELECT * FROM estudios WHERE id_number='{data_id}'")
    table_data = cursor.fetchall()
    connection.close()
    
    return table_data


def edit_db(db_table: str, id_db: int, data: dict) -> list:
    """ Edit data of a database table and returns table data updated
    
    Parameters
    ----------
    db_table: str
        Database table name
    id_db: int
        Database item id from table
    data: dict
        Data from patient or study file
    
    Returns
    -------
    table_data: list
        Data of table updated
    """
    if db_table == 'pacientes':
        last_name_value = data['last_name']
        first_name_value = data['first_name']
        id_type_value = data['id_type']
        id_value = data['id']
        birth_date_value = data['birth_date']
        sex_value = data['sex']
        weight_value = data['weight']
        weight_unit = data['weight_unit']
        height_value = data['height']
        height_unit = data['height_unit']
        bmi_value = data['bmi']
    # elif db_table == 'estudios':
    #     id_value = data['id']
    #     file_name_value = data['file_name']
    #     file_path_value = data['file_path']

    settings = QSettings(f'{sys.path[0]}/settings.ini', QSettings.Format.IniFormat)
    db_host = settings.value('db_host')
    db_port = settings.value('db_port')
    db_name = settings.value('db_name')
    db_user = settings.value('db_user')
    db_password = settings.value('db_password')
    connection = psycopg2.connect(user=db_user, 
                                  password=db_password, 
                                  host=db_host, 
                                  port=db_port, 
                                  database=db_name)
    cursor = connection.cursor()    
    
    update_query = None
    if db_table == 'pacientes':
        update_query = f"""UPDATE pacientes 
                    SET (last_name, first_name, id_type, id_number, birth_date, sex, weight, weight_unit, height, height_unit, bmi)
                    = ('{last_name_value}', '{first_name_value}', '{id_type_value}', '{id_value}', '{birth_date_value}', '{sex_value}', '{weight_value}', '{weight_unit}', '{height_value}', '{height_unit}', '{bmi_value}') 
                    WHERE id = '{id_db}' """
    # elif db_table == 'estudios':
    #     update_query = f"""UPDATE estudios 
    #                 SET (id_number, file_name, file_path)
    #                 = ('{id_value}', '{file_name_value}', '{file_path_value}') 
    #                 WHERE id = '{id_db}' """
    
    cursor.execute(update_query)
    connection.commit()

    table_data = None
    if db_table == 'pacientes':
        cursor.execute('SELECT * FROM pacientes ORDER BY id ASC')
        table_data = cursor.fetchall()
    elif db_table == 'estudios':
        cursor.execute('SELECT * FROM estudios')
        table_data = cursor.fetchall()
    
    connection.close()

    return table_data


def delete_db(db_table: str, data: str) -> list:
    """ Delete data from database table and returns table data updated
    
    Parameters
    ----------
    db_table: str
        Database table name
    data: str
        From patient: id number
        From study: study file
    
    Returns
    -------
    table_data: list
        Data of table updated
    """
    settings = QSettings(f'{sys.path[0]}/settings.ini', QSettings.Format.IniFormat)
    db_host = settings.value('db_host')
    db_port = settings.value('db_port')
    db_name = settings.value('db_name')
    db_user = settings.value('db_user')
    db_password = settings.value('db_password')
    connection = psycopg2.connect(user=db_user, 
                                  password=db_password, 
                                  host=db_host, 
                                  port=db_port, 
                                  database=db_name)
    cursor = connection.cursor()

    delete_query = None
    if db_table == 'pacientes':
        delete_query = f"DELETE FROM pacientes WHERE id_number='{data}'"
    elif db_table == 'estudios':
        delete_query = f"DELETE FROM estudios WHERE file_name='{data}'"
    cursor.execute(delete_query)
    connection.commit()

    table_data = None
    if db_table == 'pacientes':
        cursor.execute('SELECT * FROM pacientes ORDER BY id ASC')
        table_data = cursor.fetchall()
    elif db_table == 'estudios':
        cursor.execute('SELECT * FROM estudios')
        table_data = cursor.fetchall()
    
    connection.close()

    return table_data


# ----------------
# About App Dialog
# ----------------
class AboutApp(QtWidgets.QDialog):
    def __init__(self) -> None:
        """ About Me Dialog """
        super().__init__()
        # --------
        # Settings
        # --------
        self.settings = QSettings(f'{sys.path[0]}/settings.ini', QSettings.Format.IniFormat)
        self.language_value = int(self.settings.value('language'))
        self.theme_value = eval(self.settings.value('theme'))

        # ----------------
        # Generación de UI
        # ----------------
        width = 320
        height = 408
        screen_x = int(self.screen().availableGeometry().width() / 2 - (width / 2))
        screen_y = int(self.screen().availableGeometry().height() / 2 - (height / 2))

        if self.language_value == 0:
            self.setWindowTitle('Acerca de...')
        elif self.language_value == 1:
            self.setWindowTitle('About...')
        self.setGeometry(screen_x, screen_y, width, height)
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)
        self.setModal(True)
        self.setObjectName('object_about')
        if self.theme_value:
            self.setStyleSheet(f'QWidget#object_about {{ background-color: #E5E9F0;'
                f'color: #000000 }}')
        else:
            self.setStyleSheet(f'QWidget#object_about {{ background-color: #3B4253;'
                f'color: #E5E9F0 }}')


        self.about_card = mt3.Card(self, 'about_card',
            (8, 8, width-16, height-16), ('Presión Plantar', 'Plantar Pressure'), 
            self.theme_value, self.language_value)

        y, w = 48, width - 32
        mt3.FieldLabel(self.about_card, 'version_label',
            (8, y), ('Versión: 1.0', 'Version: 1.0'), self.theme_value, self.language_value)

        y += 48
        mt3.FieldLabel(self.about_card, 'desarrollado_label',
            (8, y), ('Desarrollado por:', 'Developed by:'), self.theme_value, self.language_value)

        y += 48
        mt3.IconLabel(self.about_card, 'nombre_icon',
            (8, y), 'person', self.theme_value)

        y += 6
        mt3.FieldLabel(self.about_card, 'nombre_label',
            (48, y), ('Carlos Andrés Wilches Pérez', 'Carlos Andrés Wilches Pérez'), self.theme_value, self.language_value)

        y += 30
        mt3.IconLabel(self.about_card, 'profesion_icon',
            (8, y), 'school', self.theme_value)
        
        y += 6
        mt3.FieldLabel(self.about_card, 'profesion_label',
            (48, y), ('Ingeniero Electrónico, BSc. MSc. PhD.', 'Electronic Engineer, BSc. MSc. PhD.'), self.theme_value, self.language_value)
        
        y += 24
        mt3.FieldLabel(self.about_card, 'profesion_label',
            (48, y), ('Universidad Nacional de Colombia', 'Universidad Nacional de Colombia'), self.theme_value, self.language_value)

        y += 32
        mt3.FieldLabel(self.about_card, 'profesion_label',
            (48, y), ('Maestría en Ingeniería Electrónica', 'Master in Electronic Engineering'), self.theme_value, self.language_value)

        y += 24
        mt3.FieldLabel(self.about_card, 'profesion_label',
            (48, y), ('Doctor en Ingeniería', 'Doctor in Engineering'), self.theme_value, self.language_value)

        y += 24
        mt3.FieldLabel(self.about_card, 'profesion_label',
            (48, y), ('Pontificia Universidad Javeriana', 'Pontificia Universidad Javeriana'), self.theme_value, self.language_value)

        y += 24
        mt3.IconLabel(self.about_card, 'email_icon',
            (8, y), 'mail', self.theme_value)

        y += 6
        mt3.FieldLabel(self.about_card, 'email_label',
            (48, y), ('cawilchesp@outlook.com', 'cawilchesp@outlook.com'), self.theme_value, self.language_value)

        y += 32
        self.aceptar_button = mt3.TextButton(self.about_card, 'aceptar_button',
            (w-92, y, 100), ('Aceptar', 'Ok'), 'done.png', self.theme_value, self.language_value)
        self.aceptar_button.clicked.connect(self.on_aceptar_button_clicked)

    def on_aceptar_button_clicked(self):
        self.close()

# ---------------
# About Qt Dialog
# ---------------
def about_qt_dialog(parent, language: int) -> None:
    """ About Qt Dialog """
    if language == 0:   title = 'Acerca de Qt...'
    elif language == 1: title = 'About Qt...'
    QtWidgets.QMessageBox.aboutQt(parent, title)