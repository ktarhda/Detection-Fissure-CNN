import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf

# CONFIGURATION 
st.set_page_config(page_title="Détection Fissures", page_icon="🏗️")
st.title("🏗️ Détection Fissures ")

#  RECONSTRUCTION DU MODÈLE 
def build_model():
    inputs = tf.keras.Input(shape=(227, 227, 3))
    base_model = tf.keras.applications.ResNet50(
        include_top=False, 
        weights=None, 
        input_shape=(227, 227, 3)
    )
    base_model.trainable = False
    
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)

#  CHARGEMENT DES POIDS DU MODÈLE
@st.cache_resource
def load_weights_system():
    model = build_model()
    
    if os.path.exists('modele_beton.h5'):
        try:
            model.load_weights('modele_beton.h5')
            return model
        except Exception as e:
            st.error(f"Erreur .h5 : {e}")
            return None
    elif os.path.exists('modele_resnet_final.keras'):
        try:
            model.load_weights('modele_resnet_final.keras')
            return model
        except Exception as e:
            return None
    else:
        return None

#  LANCEMENT DU SYSTÈME
with st.spinner('Démarrage...'):
    model = load_weights_system()

if model is None:
    st.error("❌ Fichier modèle introuvable !")
else:
    st.success("✅ Système Fonctionnel !")

    # INTERFACE 
    uploaded_file = st.file_uploader("Image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, width=300)
        
        if st.button('SCANNER'):
            img_resized = image.resize((227, 227))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            score = prediction[0][0]

            st.write("---")
            
            if score > 0.50:
                confiance = score * 100
                st.error(f"🚨 FISSURE DÉTECTÉE avec une precision de {confiance:.2f}%")
                st.warning("⚠️ Structure endommagée.")
            else:
                confiance = (1 - score) * 100
                st.success(f"✅ BÉTON SAIN avec une precision de {confiance:.2f}%")
                st.info("Structure en bon état.")