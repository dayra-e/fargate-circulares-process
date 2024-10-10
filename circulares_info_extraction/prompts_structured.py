# Prompts structured without anynomization

prompt_extraccion_info_oficios = [
        {'role': 'system',
         'content': """Actua como un asistente que analiza oficios provenientes de la entidad reguladora de finanzas (ASFI/Juez), para el registro de retenciones de fondos o remisiones de fondos para el banco"""},
        {'role': 'assistant',
        'content': "{input_string}"},
        {'role': 'user',
         'content': """Busca y extrae la siguiente informacion de la forma mas compacta posible.
- FECHA DE PUBLICACION DE LA CARTA CIRCULAR (En formato dd/mm/yyyy)
- NUMERO DE CARTA CIRCULAR: En formato CC-XXXX/20XX
- DOCUMENTOS QUE EMPIEZAN POR R: Una lista donde cada elemento esta en formato R-XXXX o R XXXX
NO añadas ningun otro caracter al final de cada item.
        """
         }]

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
- FECHA DE CITE:
    - siempre esta entre las primeras lineas junto a la referencia de una ciudad (por ejemplo "Cochabamba, 15 de septiembre de 2000")
    - expresar en formato dd/mm/yyyy
- NOMBRE DE LA CIUDAD DEL SOLICITANTE
- NOMBRES DEL DEMANDADO O DEMANDADOS:
    - cada demandado podria tener uno o varios nombres. Por ejemplo "Hermogenes Hernan" o "Maria Gloria Catalina"
    - si el demandado es una persona tiene apellidos. Tal vez podria tener NIT. 
    - puede haber uno o mas personas o empresas demandados
    - Si el demandado es una empresa anotale solamente en RAZON SOCIAL y deja vacio apellidos y nombres
    - Es posible que haya informacion de los demandados luego de la frase 'En cuentas de:' o 'Instruccion en cuentas de'.
    - una empresa tiene su propio NIT como TIPO DE DOCUMENTO DE IDENTIDAD
- APELLIDO MATERNO: en algunos casos podria hacer referencia al apellido del esposo (por ejemplo "de Perez") o figurar como viuda. Por ejemplo "vda. de perez".
- TIPO DE DOCUMENTO DE IDENTIDAD: solamente puede ser C.I., NIT, RUC.
- NUMERO DE DOCUMENTO DE INDENTIDAD: es una serie de digitos que va despues del TIPO DE DOCUMENTO DE IDENTIDAD.
- RAZON SOCIAL
- DEMANDANTE O DEMANDATES: 
    - Puede ser por la denuncia de una persona o mas personas, una institucion financiera o una empresa
    - el demandante puede tener apoderados.
    - Si es una institucion financiera o una empresa NO añadas su asesor/representante legal.
    - Si no es una persona anota el nombre de la institucion financiera o empresa.
- TIPO DE PROCESO: CIVIL EJECUTIVO, CIVIL MONITOREO EJECUTIVO, EJECUTIVO, PROCESO EJECUTIVO, COACTIVA, FISCAL o MONITORIO EJECUTIVO.
- MONEDA: BS, USD (dolares) o UFV.
- MONTO A SER RETENIDO: es la suma que correspondera al campo MONEDA y debe estar en formato numerico.
- DOCUMENTO DE RESPALDO: solamente puede ser NUREJ, PIET, IANUS o CUD. Dejar vacio si no se encuentra.
- NUMERO TIPO DE RESPALDO: es un numero de mas 5 digitos que corresponde al campo no nulo de DOCUMENTO DE RESPALDO.
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
Si una persona esta con apellidos tambien debe tener uno o 2 nombres.
###"""}]

prompt_extraction_tabla_info_proceso = [
            {"role": "system",
             "content": """Actua como un asistente que analiza oficios provenientes de la entidad reguladora de finanzas (ASFI/Juez), para el registro de retenciones de fondos o remisiones de fondos para el banco"""},
            {"role": "assistant",
             "content": "{input_string}"},
            {"role": "user",
             "content": """ Busca y extrae la siguiente informacion y presentala de la forma mas compacta posible en formato JSON.
- NOMBRE DE LA CIUDAD DEL SOLICITANTE corresponde a la ciudad en Bolivia de donde viene la demanda.	
- DOCUMENTO DE RESPALDO: Puede ser PLACA, POLIZA, LICENCIA, PIET, NUREJ o RESOLUCION ADMINISTRATIVA. En general es PLACA si es un documento de la aduana, dejar vacio si no se encuentra. 
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
"""},
]