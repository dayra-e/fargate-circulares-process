import json
import re
import spacy
import os

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
from presidio_analyzer.nlp_engine import NerModelConfiguration, SpacyNlpEngine
from circulares_info_extraction.config import LoadConfig

config = LoadConfig()
config.set_section('anonymization')
ANONYMIZATION_SHIFT= config.parameter('shift')
SCORE = config.parameter('score')
FIRST_ENTITY = config.parameter('first_entitity')
SECOND_ENTITY = config.parameter('second_entitity')
THIRD_ENTITY = config.parameter('third_entitity')
class Anonymizer:
    """
        Class to anonymize and deanonymize codes using a Caesar cipher.
    """
    def __init__(self, shift=ANONYMIZATION_SHIFT):
        """
        Initializes the Anonymizer class with the necessary configuration.

        Args:
            shift (int): Shift for the Caesar cipher. Default is 3.
        """
        self.shift = shift  # Desplazamiento para el cifrado César

        # Configuración del modelo spaCy personalizado
        ner_model_config = NerModelConfiguration(
            model_to_presidio_entity_mapping={
                FIRST_ENTITY: FIRST_ENTITY
            },
            labels_to_ignore=[SECOND_ENTITY, THIRD_ENTITY],
            low_score_entity_names=[],
            default_score=SCORE
        )

        # Cargar el modelo de spaCy
        ner_model_path = os.path.join(os.path.dirname(__file__), 'ner_model')
        nlp_ner = spacy.load(ner_model_path)

        # Crear una clase que herede de SpacyNlpEngine e inicializar con configuración personalizada
        class CustomSpacyNlpEngine(SpacyNlpEngine):
            def __init__(self, loaded_spacy_model, ner_model_configuration):
                super().__init__(models=[{"lang_code": "en", "model_name": "custom"}], ner_model_configuration=ner_model_configuration)
                self.nlp = {"en": loaded_spacy_model}

        self.custom_nlp_engine = CustomSpacyNlpEngine(loaded_spacy_model=nlp_ner, ner_model_configuration=ner_model_config)

        # Crear y configurar el registro de reconocedores
        registry = RecognizerRegistry()

        # Registrar el reconocedor de patrones para CODIGO
        codigo_pattern = Pattern(name="codigo", regex=r"(?<!\d)\d{6,12}(?!\d)", score=SCORE)
        codigo_recognizer = PatternRecognizer(supported_entity="CODIGO", patterns=[codigo_pattern])
        registry.add_recognizer(codigo_recognizer)

        # Inicializar AnalyzerEngine con el motor NLP personalizado y el registro de reconocedores
        self.analyzer = AnalyzerEngine(
            nlp_engine=self.custom_nlp_engine,
            registry=registry,
            supported_languages=["en"]
        )

    def caesar_cipher(self, digit, shift):
        """
        Applies the Caesar cipher to a digit.

        Args:
            digit (str): Digit to be encrypted.
            shift (int): Shift for the cipher.

        Returns:
            str: Encrypted digit.
        """
        return str((int(digit) + shift) % 10)

    def caesar_decipher(self, digit, shift):
        """
        Deciphers a digit encrypted with the Caesar cipher.

        Args:
            digit (str): Digit to be decrypted.
            shift (int): Shift used in the encryption.

        Returns:
            str: Decrypted digit.
        """    
        return str((int(digit) - shift) % 10)

    def encrypt_code(self, real_code):
        """
        Encrypts a real code using the Caesar cipher.

        Args:
            real_code (str): Real code to be encrypted.

        Returns:
            str: Encrypted code.
        """
        return ''.join(self.caesar_cipher(digit, self.shift) for digit in real_code)

    def decrypt_code(self, encrypted_code):
        """
        Decrypts a code encrypted using the Caesar cipher.

        Args:
            encrypted_code (str): Encrypted code to be decrypted.

        Returns:
            str: Decrypted code.
        """        
        return ''.join(self.caesar_decipher(digit, self.shift) for digit in encrypted_code)

    def anonymize_entities(self, text):
        """
        Anonymizes entities in a text by encrypting the found codes.

        Args:
            text (str): Text containing entities to be anonymized.

        Returns:
            tuple: Anonymized text and the analysis results of the ner_model.
        """
        results = self.analyzer.analyze(text=text, entities=["CODIGO"], language='en')
        anonymized_text = ""
        last_index = 0

        for result in sorted(results, key=lambda x: x.start):
            start = result.start
            end = result.end
            entity_text = text[start:end]

            if result.entity_type == "CODIGO":
                encrypted_code = self.encrypt_code(entity_text)
                anonymized_entity_text = f"c{encrypted_code}"

            anonymized_text += text[last_index:start] + anonymized_entity_text
            last_index = end

        anonymized_text += text[last_index:]

        return anonymized_text, results

    def deanonymize_entities(self, anonymized_json):
        """
        Deanonymizes entities in an anonymized JSON by decrypting the codes.

        Args:
            anonymized_json (dict): JSON containing anonymized entities.

        Returns:
            dict: JSON with deanonymized entities.
        """
        anonymized_text = json.dumps(anonymized_json)
        def replace_encrypted(match):
            encrypted_code = match.group(1)
            return self.decrypt_code(encrypted_code)

        patterns = {
            "COD": re.compile(r"c(\d+)")
        }

        for label, pattern in patterns.items():
            anonymized_text = pattern.sub(replace_encrypted, anonymized_text)

        return json.loads(anonymized_text)

    def deanonymize_entities_str(self, anonymized_text):
        """
        Deanonymizes entities in an anonymized text by decrypting the codes.

        Args:
            anonymized_text (str): Text containing anonymized entities.

        Returns:
            str: Text with deanonymized entities.
        """
        def replace_encrypted(match):
            encrypted_code = match.group(1)
            return self.decrypt_code(encrypted_code)

        patterns = {
            "COD": re.compile(r"c(\d+)")
        }

        for label, pattern in patterns.items():
            anonymized_text = pattern.sub(replace_encrypted, anonymized_text)

        return anonymized_text
