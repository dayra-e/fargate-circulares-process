from pydantic import BaseModel


class CircularFirstPageModel(BaseModel):
    documentos_que_empiezan_por_r: list[str]
    numero_de_carta_circular: str
    fecha_de_publicacion_de_la_carta_circular: str


class Demandado(BaseModel):
    apellido_paterno: str
    apellido_materno: str
    nombres: str
    razon_social: str
    tipo_de_documento_de_identidad: str
    numero_de_documento_de_identidad: str


class CircularStandardModel(BaseModel):
    numero_de_cite: str
    fecha_de_cite: str
    nombre_de_la_ciudad_del_solicitante: str
    demandados: list[Demandado]
    demandante: str
    tipo_de_proceso: str
    moneda: str
    monto_a_ser_retenido: str
    documento_de_respaldo: str
    numero_tipo_de_respaldo: str
    nombre_de_la_autoridad_solicitante: str
    nombre_del_juzgado: str

class CircularTablaInfoProcesoModel(BaseModel):
    nombre_de_la_ciudad_del_solicitante: str
    documento_de_respaldo: str
    numero_de_cite: str
    fecha_de_cite: str
    demandante: str
    nombre_de_la_autoridad_solicitante: str
    nombre_del_juzgado: str
    tipo_de_proceso: str
