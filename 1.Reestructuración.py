import pandas as pd

# Cargar el archivo 
df_raw = pd.read_csv('Nuevo_Dataset_Patrones_Emocionales.csv', encoding='utf-8')

# Diccionario de preguntas y su emoción correspondiente
emociones = {
    '1.  Recuerda un momento en el que lograste algo que realmente deseabas. ¿Cómo fue la experiencia y qué impacto tuvo en tu vida? ': 'Felicidad',
    '2. Describe una ocasión en la que alguien hizo algo especial o inesperado por ti y te hizo sentir bien. ¿Cómo reaccionaste? ': 'Felicidad',
    '3. Piensa en una ocasión en la que perdiste algo o a alguien importante para ti. ¿Cómo viviste ese momento y qué cambió después de ello? ': 'Tristeza',
    '4. Recuerda un día en el que sentiste que todo te salía mal. ¿Qué sucedió y cómo te sentiste al respecto? ': 'Tristeza',
    '5. Describe una situación en la que experimentaste algo que te resultó desagradable y quisiste evitar. ¿Cómo fue ese momento? ': 'Disgusto',
    '6. Recuerda una ocasión en la que viste o viviste algo que te pareció totalmente inaceptable. ¿Cómo reaccionaste y qué pensaste al respecto? ': 'Disgusto',
    '7. Piensa en un momento en el que sentiste que alguien fue injusto contigo. ¿Qué ocurrió y cómo reaccionaste? ': 'Ira',
    '8. Describe una ocasión en la que te sentiste frustrado(a) porque no tomaron en cuenta tu opinión o esfuerzo. ¿Cómo reaccionaste ante esta situación y cómo la manejaste? ': 'Ira',
    '9. Recuerda una ocasión en la que tuviste que enfrentarte a algo incierto o desconocido. ¿Cómo fue la experiencia y qué sentiste en ese momento? ': 'Miedo',
    '10. Describe un evento en el que sentiste que algo estaba fuera de tu control y no sabías qué hacer. ¿Cómo reaccionaste y qué pasó después? ': 'Miedo',
    '11. Piensa en un momento en el que ocurrió algo totalmente inesperado en tu vida. ¿Cómo fue y qué pasó después? ': 'Sorpresa',
    '12. Recuerda una ocasión en la que recibiste una noticia o viviste un evento que nunca imaginaste. ¿Cómo reaccionaste y qué impacto tuvo en ti?': 'Sorpresa'
}

# Crear un nuevo DataFrame con columnas: respuesta, emoción, pregunta
respuestas = []

for col, emocion in emociones.items():
    if col in df_raw.columns:
        for respuesta in df_raw[col].dropna():
            respuestas.append({
                'respuesta': str(respuesta).strip(),
                'emocion': emocion,
                'pregunta': col
            })

df = pd.DataFrame(respuestas)



