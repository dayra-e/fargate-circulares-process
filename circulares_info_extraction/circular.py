import pandas as pd
from api.circulares_info_extraction.config import LoadConfig

# Open AI API
config = LoadConfig('api/circulares_info_extraction/config.yml')
from api.circulares_info_extraction.utils_etl import load_tiff_pages_from_image
from api.circulares_info_extraction.image_to_text import extract_text_from_images
from api.circulares_info_extraction.text_processing import (extraer_json_info_oficios,
                                                            process_retencion_suspension_remision_in_parallel,
                                                            process_informativas_in_parallel,
                                                            process_oficios_in_parallel,
                                                            fill_carta_template,
                                                            classify_oficios,
                                                            extract_info_normativa)
from api.circulares_info_extraction.stamp_recognition import (adjusted_find_stamp_confidence,
                                                              find_stamp_pages_lambda,
                                                              separar_paginas_en_oficios)
from api.circulares_info_extraction.postprocessing import (prepare_metadata,
                                                           clean_resultados)
from api.circulares_info_extraction.judges_attribution import find_judge_name
from api.circulares_info_extraction.api_clients import openai_client
from api.circulares_info_extraction.circular_mappers import validate_dataframe


config.set_section('openai')
# Open AI
model_128k = config.parameter('model_128k')
TEMPERATURE = config.parameter('temperature')

# Lambda
config.set_section('stamp_recognition')
use_lambda = config.parameter('use_lambda')

# Image processing:
config.set_section('image_processing')
TEXTRACT_CONFIDENCE_FIRST_PAGE = config.parameter("textract_confidence_first_page")
TEXTRACT_CONFIDENCE = config.parameter("textract_confidence")


class Circular:
    def __init__(self, image_tiff, nombre_circular, tipo_circular):
        self.image_tiff = image_tiff
        self.tipo_circular = tipo_circular
        self.client = openai_client
        self.model = model_128k
        self.model_first_page = model_128k
        self.model_juzgado_search = model_128k
        self.temperature = TEMPERATURE
        self.images = []
        self.oficios = []
        self.resultados_df = pd.DataFrame()
        self.resultados_metadata = pd.DataFrame()
        self.num_paginas = 0
        self.num_oficios = 0
        self.caso = None
        self.oficios_inicios_texto = None
        self.nombre_circular = nombre_circular
        self._lambda = use_lambda

    def load_images(self):
        print("Extrayendo las imagenes de la circular")
        self.images = load_tiff_pages_from_image(self.image_tiff)
        self.imagen_info_oficios_raw = self.images[0]
        self.imagenes_oficios_raw = self.images[1:]
        self.num_paginas = len(self.imagenes_oficios_raw)

    def extract_text_first_page(self):
        print("Extraer texto de la primera pagina")
        self.info_oficios, self.imagen_info_oficios = extract_text_from_images([self.images[0]],
                                                                               confidence_threshold=TEXTRACT_CONFIDENCE_FIRST_PAGE)
        self.imagen_info_oficios = self.imagen_info_oficios[0]
        self.info_oficios = self.info_oficios[0]

    def extract_information_first_page(self):
        print("Extrayendo la informacion relevante de los oficios en la circular")
        self.resultado_info_json = extraer_json_info_oficios(self.info_oficios, self.client, self.model_first_page,
                                                             self.temperature)
        self.ids_oficios = self.resultado_info_json["DOCUMENTOS QUE EMPIEZAN POR R"]
        self.num_oficios = len(self.ids_oficios)
        print(f"Numero de oficios: {self.num_oficios}")

    def extract_text_oficios(self):
        print("Extraer texto de los oficios")
        self.paginas_oficios, self.imagenes_oficios = extract_text_from_images(self.imagenes_oficios_raw,
                                                                               confidence_threshold=TEXTRACT_CONFIDENCE)
        self.num_paginas = len(self.paginas_oficios)
        print(f"Numero de oficios: {self.num_oficios}")
        print(f"Numero de paginas: {self.num_paginas}")

    def extract_text_oficios_normativa(self):
        print("Extraer texto de los oficios")
        self.paginas_oficios, self.imagenes_oficios = extract_text_from_images(self.images,
                                                                               confidence_threshold=TEXTRACT_CONFIDENCE)
        self.num_paginas = len(self.paginas_oficios)
        print(f"Numero de oficios: {self.num_oficios}")
        print(f"Numero de paginas: {self.num_paginas}")

    def process_oficio_normativa(self):
        self.oficios_texto_normativa = "\n".join(self.paginas_oficios)
        self.tipo_circular = "normativa"
        self.resultados_df = extract_info_normativa(self.oficios_texto_normativa,
                                                    self.client,
                                                    self.model_first_page,
                                                    self.temperature)

    def determine_case_oficios(self):
        print("Determining processing strategy based on the content distribution")
        self.oficios_inicios_texto = "[]"
        # Check which case we have to handle:
        if self.num_oficios == self.num_paginas:
            print(f"Caso 1: Una pagina por oficio. {self.num_oficios} oficios")
            self.oficios = [[i] for i in self.paginas_oficios]
            self.imagenes_oficios_separados = [[i] for i in self.imagenes_oficios]
            self.caso = 1
        elif self.num_oficios == 1:
            print("Caso 2: Un solo oficio")
            self.oficios = [self.paginas_oficios]
            self.imagenes_oficios_separados = [self.imagenes_oficios]
            self.caso = 2
        else:
            print(f"Caso 3: {self.num_paginas} paginas y {self.num_oficios} oficios")
            if self._lambda:
                self.oficios_inicios = find_stamp_pages_lambda(self.imagenes_oficios_raw)
            else:
                self.min_confidence, self.oficios_inicios = adjusted_find_stamp_confidence(self.imagenes_oficios_raw,
                                                                                           self.num_oficios)
                print(f"Min confidence encontrada para este caso: {self.min_confidence}")
            self.oficios = separar_paginas_en_oficios(self.paginas_oficios, self.oficios_inicios)
            self.imagenes_oficios_separados = separar_paginas_en_oficios(self.imagenes_oficios,
                                                                         self.oficios_inicios)
            self.oficios_inicios_texto = "[" + ",".join([str(i) for i in self.oficios_inicios]) + "]"
            self.caso = 3
            print(self.oficios_inicios_texto)

    def process_oficios_retencion_suspension_remision(self):
        print(f"Processing oficios in parallel")
        # Implement the logic to process oficios in parallel based on the content distribution across pages

        self.resultados_df = process_retencion_suspension_remision_in_parallel(self.oficios,
                                                                               self.imagenes_oficios_separados,
                                                                               self.ids_oficios,
                                                                               self.resultado_info_json,
                                                                               self.client,
                                                                               self.model,
                                                                               self.temperature)

    def post_processing_oficios(self):

        self.resultados_metadata = prepare_metadata(self.resultados_df,
                                                    self.caso,
                                                    self.oficios_inicios_texto)

        self.resultados_df = clean_resultados(self.resultados_df, self.tipo_circular)

        find_judge_name(self.resultados_df, self.model_juzgado_search, self.temperature)

        self.resultados_df = self.resultados_df.merge(self.oficios_clasificacion_df,
                                                      on="IDENTIFICADOR UNICO QUE REGISTRA ASFI")

    def process_oficios_informativos(self):
        print(f"Processing oficios in parallel")
        # Implement the logic to process oficios in parallel based on the content distribution across pages

        self.resultados_df = process_informativas_in_parallel(self.oficios,
                                                              self.ids_oficios,
                                                              self.resultado_info_json,
                                                              self.client,
                                                              self.model,
                                                              self.temperature)

        self.oficios_clasificacion_df = pd.DataFrame(self.resultado_info_json['DOCUMENTOS QUE EMPIEZAN POR R'],
                                                     columns=['IDENTIFICADOR UNICO QUE REGISTRA ASFI'])
        self.oficios_clasificacion_df["TIPO DE OFICIO"] = "INFORMATIVA"

    def process_classified_oficios(self):
        self.resultados_df = process_oficios_in_parallel(self.oficios,
                                                         self.oficio_clasificacion,
                                                         self.imagenes_oficios_separados,
                                                         self.ids_oficios,
                                                         self.resultado_info_json,
                                                         self.client,
                                                         self.model,
                                                         self.temperature)

    def convert_to_cartas(self):
        cartas_resultantes = fill_carta_template(self.resultados_df)
        resultado = []
        for id_oficio, texto_carta in zip(self.ids_oficios, cartas_resultantes):
            resultado.append(f"Oficio {id_oficio}:\n{texto_carta}")
        resultados_txt = "\n".join(resultado)
        self.resultados_txt = resultados_txt

    def classify_oficios(self):
        lista_oficios_ids = self.resultado_info_json["DOCUMENTOS QUE EMPIEZAN POR R"]
        self.oficio_clasificacion = classify_oficios(self.oficios,
                                                     lista_oficios_ids,
                                                     self.client,
                                                     self.model,
                                                     self.temperature)

        self.oficios_clasificacion_df = pd.DataFrame(list(self.oficio_clasificacion.items()),
                                                     columns=['IDENTIFICADOR UNICO QUE REGISTRA ASFI',
                                                              'TIPO DE OFICIO'])

        tipos_circular = self.oficios_clasificacion_df["TIPO DE OFICIO"].unique()
        if len(tipos_circular) == 1:
            tipo_circular = tipos_circular[0]
            if tipo_circular in ["RETENCION", "SUSPENSION", "REMISION"]:
                self.tipo_circular = "retencion_suspension_remision"
            elif tipo_circular == "INFORMATIVA":
                self.tipo_circular = "informativa"
        elif len(tipos_circular) in [2, 3]:
            if len(set(tipos_circular).difference(["RETENCION", "SUSPENSION", "REMISION"])) == 0:
                self.tipo_circular = "retencion_suspension_remision"

    def process_circular_informativa(self):
        self.load_images()
        self.extract_text_first_page()
        self.extract_information_first_page()
        self.extract_text_oficios()
        self.determine_case_oficios()
        self.process_oficios_informativos()
        self.post_processing_oficios()
        self.validate_results()
        self.convert_to_cartas()

    def process_circular_retencion_suspension_remision(self):
        self.load_images()
        self.extract_text_first_page()
        self.extract_information_first_page()
        self.extract_text_oficios()
        self.determine_case_oficios()
        self.classify_oficios()
        self.process_oficios_retencion_suspension_remision()
        self.post_processing_oficios()
        self.validate_results()

    def process_circular_normativa(self):
        self.load_images()
        self.extract_text_oficios_normativa()
        self.process_oficio_normativa()
        self.validate_results()

    def classify_and_process_circular(self):
        self.load_images()
        self.extract_text_first_page()
        self.extract_information_first_page()
        if not self.ids_oficios:
            self.process_circular_normativa()
        else:
            self.process_oficios_combinados()
    def process_oficios_combinados(self):
        self.extract_text_oficios()
        self.determine_case_oficios()
        self.classify_oficios()
        self.process_classified_oficios()
        self.post_processing_oficios()
        self.validate_results()

    def validate_results(self) -> pd.io.formats.style:
        self.results_df_validated = validate_dataframe(self.tipo_circular, self.resultados_df)