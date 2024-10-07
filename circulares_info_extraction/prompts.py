# Prompts without anynomization

prompt_extraccion_info_oficios = [
        {'role': 'system',
         'content': """Actua como un asistente que analiza oficios provenientes de la entidad reguladora de finanzas (ASFI/Juez), para el registro de retenciones de fondos o remisiones de fondos para el banco"""},
        {'role': 'assistant',
        'content': """
        ###
        {input_string}
        ###"""},
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
             "content": """Actua como un asistente que analiza oficios para el registro de retenciones o suspension de retenciones
de fondos de un banco"""},
            {"role": "user",
             "content": """Lee y analiza el siguiente texto proveniente de un oficio emitido por un juzgado, enviado a la entidad reguladora de finanzas quien lo reenvia a los bancos para verificacion:
###
{input_string}
###"""},
            {"role": "user",
             "content": """Busca y extrae la siguiente informacion y presentala de la forma mas compacta posible en formato JSON.
El NOMBRE DE LA CIUDAD DEL SOLICITANTE corresponde a la ciudad en Bolivia de donde viene la demanda.	
Pueden haber 1 o mas personas o empresas demandados, encuentra todos los demandados con todos sus nombres y apellidos.
Si el demandado es una empresa anotale en RAZON SOCIAL. 
Queda atento que es posible que haya informacion de los demandados luego de la frase "En cuentas de:".
El DEMANDANTE puede ser una persona, una institucion financiera o una empresa. Si es una institucion financiera o una empresa NO añadas su asesor/representante legal. 
Anota el nombre de la institucion financiera o empresa.
El TIPO DE DOCUMENTO DE IDENTIDAD de los demandados puede ser: C.I., NIT o RUC. 
Si el tipo de documento de identidad es C.I. entonces el NUMERO DE DOCUMENTO DE IDENTIDAD es una serie de digitos y dos letras que pueden ser: LP, SC, CO, OR, PO, BE, TJ, PA, CH, TA, CB, PT.
Si el demandado es una empresa entonces, el TIPO DE DOCUMENTO DE IDENTIDAD es NIT y el NUMERO DE DOCUMENTO DE IDENTIDAD es una serie de digitos que viene despues de NIT. 
El DOCUMENTO DE RESPALDO puede ser: NUREJ, PIET o IANUS.
El NUMERO TIPO DE RESPALDO corresponde al numero de NUREJ, PIET o IANUS y es un numero de 9 digitos.
El TIPO DE PROCESO puede ser CIVIL EJECUTIVO, CIVIL MONITOREO EJECUTIVO, EJECUTIVO, COACTIVA, FISCAL o MONITOREO EJECUTIVO. 
Las fechas deben estar en formato DMY: dd/mm/yyyy.
La MONEDA puede ser BS, USD o UFV.
El MONTO A SER RETENIDO es la suma demandada y debe estar en formato numerico.
El NOMBRE DE LA AUTORIDAD SOLICITANTE se encuentra al final del documento y corresponde a la autoridad que firma el oficio.
El NOMBRE DEL JUZGADO se encuentra al principio o al final del documento y puede ser un juzgado publico civil comercial, mixto, un gobierno municipal o una aduana.
El NOMBRE DEL JUZGADO nunca es AUTORIDAD DE SUPERVISION DEL SISTEMA FINANCIERO. 
El NOMBRE DEL JUZGADO nunca es TRIBUNAL DEPARTAMENTAL DE JUSTICIA. 
El NUMERO DE CITE es una combinacion de numero de oficio y, numero de expediente o numero de causa.
Extrae todos los campos a continuacion: 
- NUMERO DE CITE (en formato OFICIO N XXX/2024 EXP. N XX/24 o RESOLUCION XX/2024) 
- FECHA DE CITE
- NOMBRE DE LA CIUDAD DEL SOLICITANTE	
- APELLIDO PATERNO DEL DEMANDADO O DEMANDADOS	
- APELLIDO MATERNO DEL DEMANDADO O DEMANDADOS	
- NOMBRES DEL DEMANDADO O DEMANDADOS	
- TIPO DE DOCUMENTO DE IDENTIDAD
- NUMERO DE DOCUMENTO DE IDENTIDAD
- RAZON SOCIAL
- DEMANDANTE O DEMANDATES
- TIPO DE PROCESO
- MONEDA 
- MONTO A SER RETENIDO
- DOCUMENTO DE RESPALDO 
- NUMERO TIPO DE RESPALDO 
- NOMBRE DE LA AUTORIDAD SOLICITANTE (Nombre de una persona)
- NOMBRE DEL JUZGADO
"""},
{"role": "user",
  "content": """Presenta los resultados en formato JSON.
Si no encuentras un valor deja el campo vacio ''.
### Ejemplo de respuesta correcta con varios demandados:
{{"NUMERO DE CITE": "OFICIO N 309/2024 CAUSA 22/2024",
"FECHA DE CITE": "16/06/2023",
"NOMBRE DE LA CIUDAD DEL SOLICITANTE": "ORURO",
"DEMANDADOS": [{{
"APELLIDO PATERNO": "MAMANI",
"APELLIDO MATERNO": "CATARI",
"NOMBRES": "SONIA",
"RAZON SOCIAL": "",
"TIPO DE DOCUMENTO DE IDENTIDAD": "C.I.",
"NUMERO DE DOCUMENTO DE IDENTIDAD": "3547170 LP"
}},
{{
"APELLIDO PATERNO": "MAMANI",
"APELLIDO MATERNO": "LOPEZ",
"NOMBRES": "INES ROXANA",
"RAZON SOCIAL": "",
"TIPO DE DOCUMENTO DE IDENTIDAD": "C.I.",
"NUMERO DE DOCUMENTO DE IDENTIDAD": "4061121 SC"
}},
{{
"APELLIDO PATERNO": "VARGAS",
"APELLIDO MATERNO": "DE JIMENEZ",
"NOMBRES": "ROCIO ALEJANDRA",
"RAZON SOCIAL": "",
"TIPO DE DOCUMENTO DE IDENTIDAD": "C.I.",
"NUMERO DE DOCUMENTO DE IDENTIDAD": "3090751 PO"
}}],
"DEMANDANTE": "SONIA PACHECO AYMA",
"TIPO DE PROCESO": "EJECUTIVO",
"MONEDA": "BS",
"MONTO A SER RETENIDO": "5,000.00",
"DOCUMENTO DE RESPALDO": "NUREJ",
"NUMERO TIPO DE RESPALDO": "204105399",
"NOMBRE DE LA AUTORIDAD SOLICITANTE": "CINTHIA FABIOLA PARDO CHAVARRIA",
"NOMBRE DEL JUZGADO": "GOBIERNO AUTONOMO MUNICIPAL DE SANTA CRUZ"}}

# Ejemplo de respuesta correcta cuando el demandado es una empresa es :
{{"NUMERO DE CITE": "OFICIO N 193/2024 EXP. 535/23",
"FECHA DE CITE": "24/11/2023",
"NOMBRE DE LA CIUDAD DEL SOLICITANTE": "ORURO",
"DEMANDADOS": [{{
"APELLIDO PATERNO": "",
"APELLIDO MATERNO": "",
"NOMBRES": "",
"RAZON SOCIAL": "SERVICE PERFECT",
"TIPO DE DOCUMENTO DE IDENTIDAD": "NIT",
"NUMERO DE DOCUMENTO DE IDENTIDAD": "193834025"
}}],
"DEMANDANTE": "FUBODE IFD",
"TIPO DE PROCESO": "COACTIVA",
"MONEDA": "BS",
"MONTO A SER RETENIDO": "30,000.00",
"DOCUMENTO DE RESPALDO": "NUREJ",
"NUMERO TIPO DE RESPALDO": "204129970",
"NOMBRE DE LA AUTORIDAD SOLICITANTE": "ABOG. IVER FERNANDO ROMERO FONTANA",
"NOMBRE DEL JUZGADO": "JUZGADO PUBLICO CIVIL Y COMERCIAL NRO 17"}}

# Ejemplo cuando el demandado es una sola persona:
{{"NUMERO DE CITE": "OFICIO N° 369/2024",
 "FECHA DE CITE": "14/05/2024",
 "NOMBRE DE LA CIUDAD DEL SOLICITANTE": "LA PAZ",
 "DEMANDADOS": [{{"APELLIDO PATERNO": "AGUILAR",
   "APELLIDO MATERNO": "CHOQUE",
   "NOMBRES": "PEDRO JUAN",
   "RAZON SOCIAL": "",
   "TIPO DE DOCUMENTO DE IDENTIDAD": "C.I.",
   "NUMERO DE DOCUMENTO DE IDENTIDAD": "6179695 LP"}}],
 "DEMANDANTE": "BANCO UNION S.A.",
 "TIPO DE PROCESO": "COACTIVA",
 "MONEDA": "BS",
 "MONTO A SER RETENIDO": "69,949.35",
 "DOCUMENTO DE RESPALDO": "NUREJ",
 "NUMERO TIPO DE RESPALDO": "2041518",
 "NOMBRE DE LA AUTORIDAD SOLICITANTE": "VICTORIA BERNAL AGULAR",
 "NOMBRE DEL JUZGADO": "JUZGADO PUBLICO CIVIL Y COMERCIAL DECIMO QUINTO DE LA CAPITAL"}}
###"""}]

prompt_extraction_table = [
    {"role": "system",
     "content": """Actua como un asistente legal que analiza tablas que contienen informacion de demandados"""},
    {"role": "user",
     "content": """Tienes que procesar una tabla proveniente de la entidad reguladora de finanzas (ASFI/Juez) que se encuentra en formato markdown.
Ciertas lineas de la tabla pueden ser ineligibles, procesa pas lineas que si son legibles.
Cada fila de esta tabla corresponde a un DEMANDADO y tiene informacion de esta persona, los nombres son nombres comunes en español. 
El TIPO DE DOCUMENTO DE IDENTIDAD solo puede ser CI, NIT o RUC.
La MONEDA puede ser BS, USD o UFV.
Para cada fila extrae la siguiente informacion:
- APELLIDO PATERNO 
- APELLIDO MATERNO
- NOMBRES	
- TIPO DE DOCUMENTO DE IDENTIDAD
- NUMERO DE DOCUMENTO DE IDENTIDAD
- RAZON SOCIAL
- MONEDA 
- MONTO A SER RETENIDO
Presenta los resultados en formato tabla con solo 8 columnas.
Procesa todas las filas.
Si no encuentras un valor deja el campo vacio ''.
Solo presenta la tabla.
Un ejemplo de respuesta correcta es :
| APELLIDO PATERNO   | APELLIDO MATERNO   | NOMBRES           | TIPO DE DOCUMENTO DE IDENTIDAD   | NUMERO DE DOCUMENTO DE IDENTIDAD   | RAZON SOCIAL   | MONEDA   | MONTO A SER RETENIDO  |
|:-------------------|:-------------------|:------------------|:---------------------------------|:-----------------------------------|:---------------|:---------|:----------------------|
| RIOJA              | IRIARTE            | MARIA LUZ         | CI                               | 5296860 CO.                        |                |    BS    |         20.5          |
| GARCIA             | SORIA              | NELY              | CI                               | 8675948 CO.                        |                |    BS    |         40.2          |
| CASTELLON          | MELGAREJO          | LUIS              | CI                               | 524460-10 CO.                      |                |    BS    |         55.2          |
|                    |                    |                   | NIT                              | 5309221017                         | EMPRESA COTAS  |    BS    |         10.5          |
        """},
    {"role": "user",
     "content": """Tabla a procesar:
###
{input_string}
###"""}]

prompt_extraction_tabla_info_proceso = [
            {"role": "system",
             "content": """Actua como un asistente legal que analiza oficios para el registro de retenciones o suspension de retenciones
de fondos para el banco"""},
            {"role": "user",
             "content": """Lee y analiza el siguiente texto proveniente de un oficio emitido por un juzgado, enviado a la entidad reguladora de finanzas quien lo reenvia a los bancos para verificacion:  
###
{input_string}
###"""},
            {"role": "user",
             "content": """ Busca y extrae la siguiente informacion y presentala de la forma mas compacta posible en formato JSON.
El NOMBRE DE LA CIUDAD DEL SOLICITANTE corresponde a la ciudad en Bolivia de donde viene la demanda.	
El DOCUMENTO DE RESPALDO puede ser: NUREJ, PIET o IANUS. 
El NUMERO TIPO DE RESPALDO corresponde al numero de NUREJ, PIET o IANUS y es un numero de 9 digitos.
El NUMERO DE CITE es una combinacion de numero de oficio y, numero de expediente o numero de causa o numero de resolucion. 
Las fechas deben estar en formato DMY: dd/mm/yyyy.
El DEMANDANTE puede ser una persona, una institucion financiera, una empresa o un gobierno municipal. Si es una institucion financiera o una empresa NO añadas su asesor/representante legal.
El NOMBRE DE LA AUTORIDAD SOLICITANTE se encuentra al final del documento y corresponde al juez o abogado que firma el oficio.
El NOMBRE DEL JUZGADO se encuentra al principio o al final del documento y puede ser un juzgado publico civil, comercial, mixto, un gobierno municipal o una aduana.
EL NOMBRE DE JUZGADO nunca es AUTORIDAD DE SUPERVISION DEL SISTEMA FINANCIERO.
El TIPO DE PROCESO puede ser CIVIL EJECUTIVO, CIVIL MONITOREO EJECUTIVO, EJECUTIVO, COACTIVA, FISCAL o MONITOREO EJECUTIVO.
Informacion a extraer. Busca todos los campos a continuacion:
- NOMBRE DE LA CIUDAD DEL SOLICITANTE
- DOCUMENTO DE RESPALDO     
- NUMERO DE TIPO DE RESPALDO 	
- NUMERO DE CITE (en formato: OFICIO N XXX/2024 EXP. N XX/24 o RESOLUCION XX/2024)
- FECHA DE CITE 
- DEMANDANTE O DEMANDATES             
- NOMBRE DE LA AUTORIDAD SOLICITANTE
- NOMBRE DEL JUZGADO
- TIPO DE PROCESO"""},
{"role": "user",
"content": """Presenta los resultados en formato JSON. Si no encuentras un valor deja el campo vacio ''.
Ejemplos de resultados correctos:
# Ejemplo 1:
{{
  "NOMBRE DE LA CIUDAD DEL SOLICITANTE": "TARIJA",
  "DOCUMENTO DE RESPALDO": "NUREJ",
  "NUMERO TIPO DE RESPALDO": "201603996",
  "NUMERO DE CITE": "OFICIO N 334/2024 EXP. 245/24",
  "FECHA DE CITE": "12/04/2024",
  "DEMANDANTE": "MAXIMA AUTORIDAD TRIBUTARIA",
  "NOMBRE DE LA AUTORIDAD SOLICITANTE": "DR. ANDRES CUEVAS GUTIERREZ",
  "NOMBRE DEL JUZGADO": "JUEZ PUBLICO DE FAMILIA NRO 10",
  "TIPO DE PROCESO": "EJECUTIVO"
}}
# Ejemplo 2:
{{
  "NOMBRE DE LA CIUDAD DEL SOLICITANTE": "LA PAZ",
  "DOCUMENTO DE RESPALDO": "NUREJ",
  "NUMERO TIPO DE RESPALDO": "1210271843",
  "NUMERO DE CITE": "N° CITE/CE/SF-DIM-72/09/2024",
  "FECHA DE CITE": "10/01/2024",
  "DEMANDANTE": "GOBIERNO MUNICIPAL DE LA PAZ",
  "NOMBRE DE LA AUTORIDAD SOLICITANTE": "MARIO GERMAN REA SALINAS",
  "NOMBRE DEL JUZGADO": "JUEZ DE INSTRUCCION PENAL NRO 8",
  "TIPO DE PROCESO": "MONITOREO EJECUTIVO"
}}"""}]

prompt_find_juzgado = [
    {"role": "system",
     "content": """Actua como un asistente legal que busca informacion en una lista que contiene juzgados."""},
    {"role": "user",
     "content": """Esta es lista de juzgados:
###{input_string1}###"""},
    {"role": "user",
     "content":  """Lee la lista y busca el nombre del juez con ciudad mas parecido a : '{input_string2}'.
Cuando busques toma en cuenta la equivalencia numerica, asi como DECIMO equivale a 10, QUINTO a 5 y  TERCERO a 3.
Ten ciudado de que las ciudades y los digitos sean los mismos.
Si no encuentras un nombre que se parezca con un alto grado de precision, entonces retorna '{input_string2}'.
Retorna unicamente el nombre encontrado en letras capitales, NINGUN OTRO SIMBOLO O PALABRA.
Ejemplos de respuestas correctas:
Ejemplo 1: JUEZ PUBLICO CIVIL Y COMERCIAL NRO 6 LA PAZ
Ejemplo 2: JUEZ PUBLICO CIVIL Y COMERCIAL NRO 1 TRINIDAD
Ejemplo 3: JUEZ DE TRABAJO Y SEGURIDAD SOCIAL NRO 3 SANTA CRUZ
Ejemplo 4: JUEZ PUBLICO MIXTO CIVIL COMERCIAL DE FAMILIA NINEZ Y ADOLECENSIA E INTRUCCION PENAL NRO 1 EL TORNO
Ejemplo 5: JUEZ PUBLICO CIVIL COMERCIAL Y DE SENTENCIA PENAL NRO 1 COMARAPA COCHABAMBA
Ejemplo 6: JUEZ DE PARTIDO LIQUIDADOR Y SENTENCIA NRO 4 EL ALTO"""}]

prompt_find_judge = [
    {'role': 'system',
     'content': """Actua como un asistente legal que busca un nombre en una lista que contiene nombres de jueces."""},
    {'role': 'user',
     'content':
"""Esta es lista:
###{input_string1}###"""},
    {'role': 'user',
     'content':  """Lee la lista completa y encuentra el nombre mas parecido a: '{input_string2}'.
Si no encuentras un nombre que se parezca con un alto grado de precision, entonces retorna '{input_string2}'.
Retorna unicamente el valor en letras capitales, NINGUN OTRO SIMBOLO O PALABRA.
# Ejemplos de respuestas correctas:
Ejemplo 1: ABOG. JORGE VICENTE OROPEZA MONTECINOS 
Ejemplo 2: LIC. BETTY NOGALES BOHORQUEZ 
Ejemplo 3: SANDRA GLADYS ALDAYUZ AVILES 
Ejemplo 4: DRA. CARLA FABIOLA CORIA PRIETO"""}]

prompt_informativas_carta = [
            {'role': 'system',
             'content': """Actua como un asistente legal que analiza oficios informativos para el departamento de operaciones de un banco"""},
            {'role': 'user',
             'content':
"""Lee y analiza el siguiente texto proveniente de un una carta de instruccion emitido por un juzgado, enviado a la entidad reguladora de finanzas 
quien lo comunica a los bancos para que lleven a cabo cierta instruccion:
###
{input_string}
###"""},
            {'role': 'user',
             'content': """El banco tiene que responder a esta carta con la instruccion requerida despues de haber verificado en sus sistemas.
Para realizar esta carta tienes que encontrar la siguiente informacion:
- CIUDAD de donde viene la carta
- FECHA DE PUBLICACION de la carta (en formato literal)
- JUEZ
- JUZGADO
- DEMANDANTES
- DEMANDADOS
- TIPO DE PROCESO
- DOCUMENTO DE RESPALDO (Solo puede ser NUREJ, PIET, IANUS)
- NUMERO DOCUMENTO DE RESPALDO 
- INSTRUCCION ESPECIFICA PARA EL BANCO

Escribe la carta de respuesta del banco. 
Solo reemplaza los campos en {{}}, no cambies los campos que estan marcados con XXX en el Resultado.
Resultado:
###
{{CIUDAD}}, XXX
COS/REQ/XXXX/2024

Señora
{{JUEZ}}
{{JUZGADO}}
DE LA CIUDAD DE {{CIUDAD}} 
Presente. –

De nuestra consideración:

Dando cumplimiento a la Carta Circular XXX emitida por la Autoridad de Supervisión del Sistema Financiero (ASFI), de fecha XXX, 
dentro del proceso {{TIPO DE PROCESO}} seguido por {{DEMANDANTES}} en contra de {{DEMANDADOS}}, 
caso {{DOCUMENTO DE RESPALDO}}: {{NUMERO DOCUMENTO DE RESPALDO}}, su oficio de fecha {{FECHA DE PUBLICACION}}, informamos lo siguiente: 

- 	{{INSTRUCCION ESPECIFICA PARA EL BANCO}}

Con este motivo, saludamos a usted con nuestras consideraciones más distinguidas.

Atentamente, 
Banco ###
"""}]

prompt_informativas = [
            {'role': 'system',
             'content': """Actua como un asistente legal que analiza oficios informativos para el departamento de operaciones de un banco"""},
            {'role': 'user',
             'content':
"""Lee y analiza el siguiente texto proveniente de un una carta de instruccion emitido por un juzgado, enviado a la entidad reguladora de finanzas 
quien lo comunica a los bancos para que lleven a cabo cierta instruccion:
###
{input_string}
###"""},
            {'role': 'user',
             'content': """El banco tiene que responder a esta carta con la instruccion requerida.
Para realizar esta carta tienes que encontrar la siguiente informacion:
- NOMBRE DE LA CIUDAD DEL SOLICITANTE (ciudad de donde viene la carta)
- FECHA DE PUBLICACION de la carta (En formato dd/mm/yyyy)
- NOMBRE DE LA AUTORIDAD SOLICITANTE (Nombre del Juez)
- NOMBRE DEL JUZGADO (Puede ser un juzgado público, mixto, comercial de familia, adolescencia, de trabajo, seguridad social, de sentencia penal, de instruccion penal, y tiene un numero asignado)
- DEMANDANTES
- DEMANDADOS
- TIPO DE PROCESO
- TIPO DE CASO (Solo puede ser NUREJ, PIET, IANUS)
- NUMERO DOCUMENTO DE RESPALDO 
- INSTRUCCION ESPECIFICA PARA EL BANCO
Presenta los resultados en formato JSON:
Ejemplo de una respuesta correcta:
{{
"NOMBRE DE LA CIUDAD DEL SOLICITANTE":"Beni",
"FECHA DE PUBLICACION":"12/04/2024",
"NOMBRE DE LA AUTORIDAD SOLICITANTE":"Raisa Arellano Teran",
"NOMBRE DEL JUZGADO":"JUZGADO PUBLICO CIVIL Y COMERCIAL Y DE FAMILIA NRO 9",
"DEMANDANTES":"FANNY ROSARIO PANIAGUA VERA",
"DEMANDADOS":"JOSE ALFREDO PANIAGUA VERA",
"TIPO DE PROCESO":"Demanda de declaración de interdicción",
"TIPO DE CASO":"NUREJ",
"NUMERO DOCUMENTO DE RESPALDO ":"8078378",
"INSTRUCCION ESPECIFICA PARA EL BANCO":"Verificar si el señor PAULO MITSUO MAKI PANIAGUA con C.I. 10848536 mantiene cuentas bancarias en la institucion"
}}
Ejemplo de una respuesta correcta con varios demandados y demandantes:
{{
"NOMBRE DE LA CIUDAD DEL SOLICITANTE":"La Paz",
"FECHA DE PUBLICACION":"15/04/2024",
"NOMBRE DE LA AUTORIDAD SOLICITANTE":"ABG. JOSSELYN M. MELCHOR HUARACHI",
"NOMBRE DEL JUZGADO":"JUZGADO PUBLICO CIVIL Y COMERCIAL Y DE FAMILIA NRO 9",
"DEMANDANTES":"FANNY ROSARIO PANIAGUA VERA y JUAN PABLO GOMEZ PARDO",
"DEMANDADOS":"JOSE ALFREDO PANIAGUA VERA, PEDRO HUANCA CHOQUE y MARIA ANDREA CALVO TICONA",
"TIPO DE PROCESO":"Investigación por denuncia",
"TIPO DE CASO":"IANUS",
"NUMERO DOCUMENTO DE RESPALDO ":"1208317322",
"INSTRUCCION ESPECIFICA PARA EL BANCO":"Remitir informe o certificación sobre las cuentas de ahorro y crédito activos de la ciudadana: Karina Coraite Fajardo con C.I. N° 6391210. 2. Remitir informe o certificación sobre las cuentas de ahorro y crédito activos del ciudadano: Jose Flores Romero con C.I. N° 1438096"
}}
###
"""}]


prompt_clasificar_oficio =  [
        {'role': 'system',
         'content': """Actua como un asistente legal que clasifica una carta de oficio"""},
        {'role': 'user',
         'content': """Lee el siguiente TEXTO que corresponde a un oficio:
###
{input_string}
###"""},
        {'role': 'user',
         'content': """
Esta carta corresponde a un tipo de oficio que puede ser RETENCION, SUSPENSION, REMISION o INFORMATIVA.
RETENCION: Una carta de retencion es una demanda de retencion o congelamiento de fondos.
SUSPENSION: Una carta de suspension se refiere a una demanda de suspension o levantamiento de retencion de fondos o tambien puede ser descongelamiento o desbloqueo de fondos.
REMISION: Una carta de remision se refiere a remitir, tranferir o enviar un monto de dinero a la persona o empresa en cuestion.
INFORMATIVA: Es una carta de instruccion a las autoridades correspondinetes que requiere informacion financiera de una persona o empresa.
Tu respuesta debe ser solamente RETENCION, SUSPENSION, REMISION o INFORMATIVA, no añadas ningun otro texto mas.
"""}]



prompt_extraction_normativa =  [
        {'role': 'system',
         'content': """Actua como un asistente legal que extrae informacion de una carta de normativa que envia la autoridad de regulacion de finanzas ASFI a los bancos"""},
        {'role': 'user',
         'content':
"""Lee el siguiente TEXTO que corresponde a esta carta:
###
{input_string}
###"""},
    {'role': 'user',
     'content': """Busca y extrae la siguiente informacion de la forma mas compacta posible.
        - FECHA DE PUBLICACION DE LA CARTA CIRCULAR (En formato dd/mm/yyyy)
        - NUMERO DE CARTA CIRCULAR: En formato CC-XXXX/2024 
        - NUMERO DE TRAMITE: El Numero de tramite que se encuentra cerca de la REF:
        - RESUMEN: Resumen ejecutivo del contenido del oficio, corto y conciso, maximo en 3 oraciones y no en formato carta.
        - REFERENCIA: Resumen del contenido en una oracion y en formato titulo"""
     },
    {'role': 'user',
     'content': """Presenta los resultados en formato JSON.
        ### Ejemplo de resultado correcto: 
         {{
            "FECHA DE PUBLICACION DE LA CARTA CIRCULAR": "05/03/2024",
            "NUMERO DE CARTA CIRCULAR": "CC-7143/2024",
            "NUMERO DE TRAMITE":"T-2009281921",
            "RESUMEN": "La información del Servicio de Impuestos Nacionales, Aduana Nacional y la Autoridad de Fiscalización del Juego ha sido transmitida a través del Sistema SIREFO. Las Entidades de Intermediación Financiera y del Mercado de Valores deben cumplir con las instrucciones recibidas. Los resultados del cumplimiento deben ser comunicados a la Autoridad Administrativa correspondiente.",
            "REFERENCIA": "ACTUALIZACIÓN DEL SISTEMA DE CAPTURA DE INFORMACIÓN PERIÓDICA"
         }}###"""
     }]