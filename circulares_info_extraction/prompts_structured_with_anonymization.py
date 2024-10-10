# Prompts structured with anynomization

prompt_extraction_standard = [
            {"role": "system",
             "content": """Actua como un asistente que analiza oficios provenientes de la entidad reguladora de finanzas (ASFI/Juez), para el registro de retenciones de fondos o remisiones de fondos para el banco"""},
            {"role": "assistant",
             "content": "{input_string}"},
            {"role": "user",
             "content": """Busca y extrae la siguiente informacion:
- NUMERO DE CITE:
    - de varias lineas y es la combinacion de diferentes formatos, de ser posible en el siguiente orden: numero de Oficio (Of. N XX/20XX, OFICIO N XXX/20XX), numero de Expediente (XX/20XX, EXP. N XX/XXX), Resolucion (XX/20XX) o numero de causa.
    - podria tener el formato OFICIO N XXX/20XX EXP. N XX/XX o RESOLUCION XX/20XX)
- FECHA DE CITE: en formato dd/mm/yyyy
- NOMBRE DE LA CIUDAD DEL SOLICITANTE
- NOMBRES DEL DEMANDADO O DEMANDADOS:
    - cada demandado podria tener uno o varios nombres. Por ejemplo "Hermogenes Hernan" o "Maria Gloria Catalina"
    - si el demandado es una persona tiene apellidos y nombres. Tal vez podria tener NIT. 
    - puede haber uno o mas personas o empresas demandados
    - Si el demandado es una empresa anotale solamente en RAZON SOCIAL y deja vacio apellidos y nombres
    - Es posible que haya informacion de los demandados luego de la frase 'En cuentas de:' o 'Instruccion en cuentas de'.
    - una empresa tiene su propio NIT como TIPO DE DOCUMENTO DE IDENTIDAD
- APELLIDO MATERNO: podria hacer referencia al apellido del esposo o figurar como viuda (vda).
- TIPO DE DOCUMENTO DE IDENTIDAD: solamente puede ser C.I., NIT, RUC.
- NUMERO DE DOCUMENTO DE INDENTIDAD: 
    - Si el tipo de documento de identidad es C.I. entonces el NUMERO DE DOCUMENTO DE IDENTIDAD es una serie de digitos  que tienen c por delante y 
    dos letras que pueden ser: LP, SC, CO, OR, PO, BE, TJ, PA, CH, TA, CB, PT, por ejemplo "c3547170 SC" o "c193834025".
- RAZON SOCIAL
- DEMANDANTE O DEMANDATES: 
    - Puede ser por la denuncia de una persona o mas personas, una institucion financiera o una empresa
    - el demandante puede tener apoderados.
    - Si es una institucion financiera o una empresa NO añadas su asesor/representante legal.
    - Si no es una persona anota el nombre de la institucion financiera o empresa.
- TIPO DE PROCESO: CIVIL EJECUTIVO, CIVIL MONITOREO EJECUTIVO, EJECUTIVO, PROCESO EJECUTIVO, COACTIVA, FISCAL o MONITORIO EJECUTIVO.
- MONEDA: BS, USD (dolares) o UFV.
- MONTO A SER RETENIDO: es la suma que correspondera al campo MONEDA y debe estar en formato numerico.
- DOCUMENTO DE RESPALDO: solamente puede ser NUREJ, PIET, IANUS o CUD.
- NUMERO TIPO DE RESPALDO: es el numero del DOCUMENTO DE RESPALDO de mas 5 digitos que tienen c por delante.
- NOMBRE DE LA AUTORIDAD SOLICITANTE: 
    - nombre de una persona juez abogado u otro, que se encuentra al final del documento y corresponde a la autoridad que firma el oficio.
    - nunca es una institucion.
- NOMBRE DEL JUZGADO:
    - se encuentra al principio o al final del documento y puede ser un juzgado publico civil comercial, mixto, un gobierno municipal o una aduana.
    - nunca es AUTORIDAD DE SUPERVISION DEL SISTEMA FINANCIERO.
    - nunca es TRIBUNAL DEPARTAMENTAL DE JUSTICIA.
Si el TIPO DE DOCUMENTO DE IDENTIDAD es C.I. entonces el NUMERO DE DOCUMENTO DE INDENTIDAD podria ser una serie de digitos y dos letras que pueden ser: LP, SC, CO, OR, PO, BE, TJ, PA, CH, TA, CB, PT. Por ejemplo "123456-1G SC"
Si el demandado es una empresa entonces, el TIPO DE DOCUMENTO DE IDENTIDAD es NIT y el NUMERO DE DOCUMENTO DE IDENTIDAD es una serie de digitos que viene despues de NIT.
El demandante y sus apoderados no pueden ser parte de los demandados.
Convierte todo el texto en mayusculas.
Si no encuentras un valor deja el campo vacio ''.
IMPORTANTE: si cualquier codigo tiene c adelante colocalo con la c (ejemplo: c01345678)
###"""}]

prompt_extraction_tabla_info_proceso = [
            {"role": "system",
             "content": """Actua como un asistente que analiza oficios provenientes de la entidad reguladora de finanzas (ASFI/Juez), para el registro de retenciones de fondos o remisiones de fondos para el banco"""},
            {"role": "assistant",
             "content": "{input_string}"},
            {"role": "user",
             "content": """ Busca y extrae la siguiente informacion y presentala de la forma mas compacta posible en formato JSON.
- NOMBRE DE LA CIUDAD DEL SOLICITANTE corresponde a la ciudad en Bolivia de donde viene la demanda.	
- DOCUMENTO DE RESPALDO: solamente puede ser NUREJ, PIET, IANUS o CUD. Dejar vacio si no se encuentra.
- NUMERO TIPO DE RESPALDO: es un numero de mas 5 digitos que corresponde al campo no nulo de DOCUMENTO DE RESPALDO. 
- NUMERO DE CITE:
    - de varias lineas y es la combinacion de diferentes formatos, de ser posible en el siguiente orden: numero de Oficio (Of. N XX/20XX, OFICIO N XXX/20XX), numero de Expediente (XX/20XX, EXP. N XX/XXX), Resolucion (XX/20XX) o numero de causa.
    - podria tener el formato OFICIO N XXX/20XX EXP. N XX/XX o RESOLUCION XX/20XX)
- FECHA DE CITE: 
    - siempre esta entre las primeras lineas junto a la referencia de una ciudad (por ejemplo "Cochabamba, 15 de septiembre de 2000")
    - expresar en formato dd/mm/yyyy
- DEMANDANTE O DEMANDATES: 
    - Puede ser por la denuncia de una persona o mas personas, una institucion financiera o una empresa
    - el demandante puede tener apoderados.
    - Si es una institucion financiera o una empresa NO añadas su asesor/representante legal.
    - Si no es una persona anota el nombre de la institucion financiera o empresa.
- NOMBRE DE LA AUTORIDAD SOLICITANTE: 
    - nombre de una persona juez abogado u otro, que se encuentra al final del documento y corresponde a la autoridad que firma el oficio.
    - nunca es una institucion.
- NOMBRE DEL JUZGADO:
    - se encuentra al principio o al final del documento y puede ser un juzgado publico civil comercial, mixto, un gobierno municipal o una aduana.
    - nunca es AUTORIDAD DE SUPERVISION DEL SISTEMA FINANCIERO.
    - nunca es TRIBUNAL DEPARTAMENTAL DE JUSTICIA.
- TIPO DE PROCESO: CIVIL EJECUTIVO, CIVIL MONITOREO EJECUTIVO, EJECUTIVO, PROCESO EJECUTIVO, COACTIVA, FISCAL o MONITORIO EJECUTIVO.
Convierte todo el texto en mayusculas.
Si no se encuentra en el texto el valor de algun campo dejarlo vacio ''.
IMPORTANTE: si cualquier codigo tiene c adelante colocalo con la c (ejemplo: c01345678)
"""},
]