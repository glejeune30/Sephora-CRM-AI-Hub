import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION ET DESIGN ---
st.set_page_config(page_title="Sephora AI Hub", layout="wide")



# --- 2. IMPORTATION DES DONNÉES ---
if 'df' not in st.session_state:
    st.markdown("<h2 style='text-align: center;'>💄 Sephora AI Intelligence Hub</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Importez votre base de données pour initialiser les moteurs</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Axe_Desc' in df.columns:
            df['Axe_Desc'] = df['Axe_Desc'].replace({'MAEK UP': 'MAKE UP'})
        rfm_mapping = {1: "1 - VIP", 2: "2 - Good Customer", 3: "3 - Opportunist", 4: "4 - New 3M"}
        if 'RFM_Segment_ID' in df.columns:
            df['RFM_Name'] = df['RFM_Segment_ID'].map(rfm_mapping)
        
        st.session_state['df'] = df
        st.rerun() 

else:
    df = st.session_state['df']
    
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("<h2 style='margin-top:0;'>💄 Sephora AI Hub</h2>", unsafe_allow_html=True)
    with col_btn:
        if st.button("🔄 Changer de fichier", use_container_width=True):
            del st.session_state['df']
            st.rerun()

    st.markdown("---")
    # === CRÉATION DES SIX ONGLETS ===
 # === CRÉATION DU MENU PRINCIPAL (10 ONGLETS) ===
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🏠 Accueil", 
        "🎯 Moteur d'Affinité", 
        "🧬 Labo Segmentation", 
        "📊 Fiche Marque",
        "🚀 Acquisition",
        "👤 Personas",
        "👨‍👩‍👧‍👦 Analyse Générations", # LE NOUVEL ONGLET
        "🔮 Prédiction LTV",
        "💸 Simulateur R.O.I",
        "🔎 Audit Data"
    ])
    # ==========================================
    # ONGLET 1 : EXECUTIVE COCKPIT (PAGE D'ACCUEIL)
    # ==========================================
    with tab1:
        st.markdown("<h3 style='text-align: center;'>Bienvenue dans le Sephora CRM Hub 🚀</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 30px;'>L'Intelligence Artificielle au service de la Fidélisation et de la Croissance.</p>", unsafe_allow_html=True)
        
        # Calcul des Super-KPIs
        ca_total = df['salesVatEUR'].sum()
        clients_actifs = df['anonymized_card_code'].nunique()
        tickets_totaux = df['anonymized_Ticket_ID'].nunique()
        panier_moyen_global = ca_total / tickets_totaux if tickets_totaux > 0 else 0
        
        # Affichage des métriques principales
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="sephora-card" style="border-top: 4px solid #000; text-align: center;">
            <div class="metric-title">Chiffre d'Affaires Analysé</div>
            <div class="metric-value" style="font-size: 32px;">{ca_total:,.0f} €</div>
        </div>""", unsafe_allow_html=True)
        
        c2.markdown(f"""
        <div class="sephora-card" style="border-top: 4px solid #000; text-align: center;">
            <div class="metric-title">Base Clients Active</div>
            <div class="metric-value" style="font-size: 32px;">{clients_actifs:,}</div>
        </div>""", unsafe_allow_html=True)
        
        c3.markdown(f"""
        <div class="sephora-card" style="border-top: 4px solid #E50000; text-align: center;">
            <div class="metric-title">Panier Moyen Global</div>
            <div class="metric-value" style="font-size: 32px;">{panier_moyen_global:.2f} €</div>
        </div>""", unsafe_allow_html=True)
        
        # Résumé de la mission
        st.markdown("""
        <div class="sephora-card">
            <h4>🎯 Mission du Dashboard</h4>
            <p>Ce cockpit interactif répond au <b>Use Case n°3 : "Brand Affinity & Cold Start"</b>. Notre objectif est de dépasser la simple analyse du panier (Market Basket) en intégrant le niveau de maturité du client (RFM) pour personnaliser les recommandations.</p>
            <ul>
                <li><b>Pour le CRM :</b> Utiliser le <i>Moteur d'Affinité</i> pour cibler les bonnes campagnes sur les bonnes personnes.</li>
                <li><b>Pour le Marketing :</b> Utiliser les <i>Personas</i> et le <i>Labo de Segmentation</i> pour comprendre la psychologie d'achat.</li>
                <li><b>Pour la Direction :</b> Utiliser le <i>Simulateur ROI</i> pour quantifier l'impact de nos modèles algorithmiques.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    # ==========================================
    # ONGLET 2 : MOTEUR D'AFFINITÉ (Le retour du graphique)
    # ==========================================
    with tab2:
        # --- LA MATRICE STRATÉGIQUE (Corrigée et Lisible) ---
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**1. Matrice de Positionnement Stratégique des Marques**")
        st.write("Cartographie du catalogue Sephora selon la loyauté moyenne des acheteurs et la valeur générée. Seules les 15 plus grandes marques sont affichées en texte pour la lisibilité (survolez les autres bulles pour voir les détails).")
        
        with st.spinner("Génération de la matrice des marques..."):
            if 'RFM_Segment_ID' in df.columns:
                # 1. Calculs
                brand_matrix = df.groupby('brand').agg(
                    avg_rfm=('RFM_Segment_ID', 'mean'),
                    avg_spend=('salesVatEUR', 'mean'),
                    clients_uniques=('anonymized_card_code', 'nunique')
                ).reset_index()
                
                # 2. Filtre anti-bruit (On enlève les micro-marques qui faussent l'axe)
                brand_matrix = brand_matrix[brand_matrix['clients_uniques'] > 50]
                
                # 3. Étiquetage intelligent (Seulement le Top 15 en texte visible)
                top_brands = brand_matrix.nlargest(15, 'clients_uniques')['brand'].tolist()
                brand_matrix['label'] = brand_matrix['brand'].apply(lambda x: x if x in top_brands else "")
                
                # 4. Le Graphique Plotly
                fig_matrix = px.scatter(
                    brand_matrix, x='avg_rfm', y='avg_spend', 
                    size='clients_uniques', color='avg_spend',
                    hover_name='brand', text='label',
                    color_continuous_scale=['#000000', '#E50000'],
                    labels={'avg_rfm': "<- VIPs (1) | Loyauté Moyenne | Nouveaux (4) ->", 
                            'avg_spend': "Dépense Moyenne (€)"},
                    size_max=40
                )
                
                # Design
                fig_matrix.update_traces(textposition='top center', textfont_size=11, textfont_color='#E50000')
                fig_matrix.update_layout(plot_bgcolor='white', coloraxis_showscale=False, height=500, margin=dict(t=10, l=0, r=0, b=0))
                fig_matrix.update_xaxes(autorange="reversed", showgrid=True, gridcolor='lightgrey')
                fig_matrix.update_yaxes(showgrid=True, gridcolor='lightgrey')
                
                st.plotly_chart(fig_matrix, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        col_filtres, col_resultats = st.columns([1, 2.5], gap="large")
        
        with col_filtres:
            st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
            st.markdown("**1. Définir l'Audience Cible**")
            liste_segments = sorted(df['RFM_Name'].dropna().unique().tolist())
            liste_categories = sorted(df['Axe_Desc'].dropna().unique().tolist())
            
            segment = st.selectbox("Segment RFM cible :", liste_segments)
            categorie = st.selectbox("Catégorie d'intérêt :", liste_categories)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_resultats:
            st.markdown("**2. Analyse d'Affinité (Algorithme de Lift)**")
            
            df_cat = df[df['Axe_Desc'] == categorie]
            
            if not df_cat.empty:
                global_share = df_cat['brand'].value_counts(normalize=True)
                df_seg = df_cat[df_cat['RFM_Name'] == segment]
                segment_share = df_seg['brand'].value_counts(normalize=True)
                
                affinity = pd.DataFrame({'Global': global_share, 'Segment': segment_share}).dropna()
                affinity = affinity[affinity['Segment'] > 0.01] 
                affinity['Lift'] = affinity['Segment'] / affinity['Global']
                
                top_brands = affinity.sort_values(by='Lift', ascending=False)
                
                if not top_brands.empty:
                    c1, c2, c3 = st.columns(3)
                    def display_card(col, rank, brand_name, lift_val, segment_share_val):
                        border = "border-top: 4px solid #E50000;" if rank == 1 else "border: 1px solid #E0E0E0;"
                        bg_color = "#000000" if rank == 1 else "#FFFFFF"
                        text_color = "#FFFFFF" if rank == 1 else "#000000"
                        title_color = "#AAAAAA" if rank == 1 else "#888888"
                        
                        col.markdown(f"""
                        <div class="sephora-card" style="background-color: {bg_color} !important; {border} padding: 15px;">
                            <div class="metric-title" style="color: {title_color} !important;">Choix n°{rank}</div>
                            <div class="metric-value" style="color: {text_color} !important;">{brand_name}</div>
                            <div style="color: {text_color} !important;">Affinité (Lift) : <span class="badge-green">x{lift_val:.2f}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(top_brands) > 0: display_card(c1, 1, top_brands.index[0], top_brands.iloc[0]['Lift'], top_brands.iloc[0]['Segment'])
                    if len(top_brands) > 1: display_card(c2, 2, top_brands.index[1], top_brands.iloc[1]['Lift'], top_brands.iloc[1]['Segment'])
                    if len(top_brands) > 2: display_card(c3, 3, top_brands.index[2], top_brands.iloc[2]['Lift'], top_brands.iloc[2]['Segment'])
                    
                    # LE GRAPHIQUE PLOTLY (Clair et lisible)
                    plot_data = top_brands.head(5).reset_index()
                    if 'brand' in plot_data.columns: plot_data = plot_data.rename(columns={'brand': 'Marque'})
                    elif 'index' in plot_data.columns: plot_data = plot_data.rename(columns={'index': 'Marque'})

                    fig = px.bar(plot_data, x='Marque', y=['Segment', 'Global'], barmode='group',
                                 title=f"Preuve Analytique : Sur-performance sur le segment '{segment}'",
                                 labels={'value': 'Part de Marché', 'variable': 'Cohorte (Segment vs Global)'},
                                 color_discrete_sequence=['#E50000', '#000000'])
                    fig.update_layout(plot_bgcolor='white', margin=dict(t=40, l=0, r=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Pas assez de données pour ce segment.")

    # ==========================================
    # ONGLET 3 : LABO DE SEGMENTATION (K-Means Expliqué)
    # ==========================================
    with tab3:
        col_k, col_plot = st.columns([1, 2.5], gap="large")
        
        with col_k:
            st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
            st.markdown("**Paramétrage de l'IA**")
            st.write("L'algorithme regroupe les clients selon leurs comportements réels, sans a priori.")
            k_clusters = st.slider("Nombre de profils à générer (K) :", min_value=2, max_value=5, value=3)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_plot:
            with st.spinner("Analyse K-Means en cours..."):
                # 1. Préparation (Agrégation par client)
                customer_df = df.groupby('anonymized_card_code').agg(
                    Depense_Totale=('salesVatEUR', 'sum'),
                    Frequence_Achat=('anonymized_Ticket_ID', 'nunique')
                ).reset_index()
                
                customer_df = customer_df[(customer_df['Depense_Totale'] > 0) & (customer_df['Depense_Totale'] < 2000)]
                
                # 2. Modèle Machine Learning
                features = customer_df[['Depense_Totale', 'Frequence_Achat']]
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                customer_df['Cluster_ID'] = kmeans.fit_predict(features_scaled)
                customer_df['Cluster_Name'] = "Profil " + customer_df['Cluster_ID'].astype(str)
                
                # 3. LE DÉCRYPTAGE (Ce qui manquait pour comprendre)
                st.markdown("**1. Décryptage des profils identifiés par l'IA (Centroids)**")
                st.write("Voici les caractéristiques moyennes des groupes que l'algorithme vient de créer :")
                
                summary_df = customer_df.groupby('Cluster_Name').agg(
                    Nombre_de_Clients=('anonymized_card_code', 'count'),
                    Panier_Moyen_Global=('Depense_Totale', 'mean'),
                    Frequence_Moyenne=('Frequence_Achat', 'mean')
                ).round(1).reset_index()
                
                # Affichage d'un beau tableau explicatif
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # 4. LE GRAPHIQUE VISUEL
                st.markdown("**2. Visualisation de la distribution**")
                fig2 = px.scatter(customer_df, x='Frequence_Achat', y='Depense_Totale', color='Cluster_Name',
                                 labels={'Frequence_Achat': 'Fréquence (Nb de tickets)', 'Depense_Totale': 'Dépense Totale Cumulée (€)'},
                                 color_discrete_sequence=px.colors.qualitative.Bold,
                                 opacity=0.7)
                fig2.update_layout(plot_bgcolor='white', margin=dict(t=10, l=0, r=0, b=0))
                st.plotly_chart(fig2, use_container_width=True)
                 
    # ==========================================
    # ONGLET 4 : FICHE MARQUE (Scorecard & Corrélation)
    # ==========================================
    with tab4:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**1. Sélectionnez une marque à analyser :**")
        liste_marques = sorted(df['brand'].dropna().unique().tolist())
        selected_brand = st.selectbox("", liste_marques, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        if selected_brand:
            df_brand = df[df['brand'] == selected_brand]
            
            # --- 1. CALCULS DE BASE ---
            ca_total = df_brand['salesVatEUR'].sum()
            tickets_uniques = df_brand['anonymized_Ticket_ID'].unique()
            nb_tickets = len(tickets_uniques)
            panier_moyen = ca_total / nb_tickets if nb_tickets > 0 else 0
            
            # --- 2. CALCUL DES TOPS (Lieu & Catégorie) - La correction est ici ! ---
            top_store = df_brand.groupby('store_city')['salesVatEUR'].sum().idxmax() if not df_brand.empty else "N/A"
            top_category = df_brand.groupby('Axe_Desc')['salesVatEUR'].sum().idxmax() if not df_brand.empty else "N/A"
            
            # --- 3. ANALYSE DE CORRÉLATION (PANIER) ---
            paniers_contenant_marque = df[df['anonymized_Ticket_ID'].isin(tickets_uniques)]
            autres_produits_panier = paniers_contenant_marque[paniers_contenant_marque['brand'] != selected_brand]
            
            if not autres_produits_panier.empty:
                corrélation_stats = autres_produits_panier['brand'].value_counts()
                top_companion_brand = corrélation_stats.idxmax()
                nb_co_occurrence = corrélation_stats.max()
                taux_attachement = (nb_co_occurrence / nb_tickets) * 100
            else:
                top_companion_brand = "Aucune"
                taux_attachement = 0

            # --- 4. AFFICHAGE DES KPIS EN TABLEAU PROPRE ---
            st.markdown("**2. Fiche d'Identité & Performances**")
            
            scorecard_data = {
                "Indicateur Clé": [
                    "💰 Chiffre d'Affaires Total",
                    "🛍️ Panier Moyen",
                    "📍 Point de Vente N°1 (CA)",
                    "🏷️ Catégorie Phare",
                    "🔥 Produit le plus acheté avec (Corrélation)"
                ],
                "Résultat Analytique": [
                    f"{ca_total:,.0f} €".replace(',', ' '),
                    f"{panier_moyen:.2f} €",
                    top_store,
                    top_category,
                    f"{top_companion_brand} (Présent dans {taux_attachement:.1f}% des paniers)"
                ]
            }
            
            df_scorecard = pd.DataFrame(scorecard_data)
            st.dataframe(df_scorecard, use_container_width=True, hide_index=True)
            
            # --- 5. GRAPHIQUE RFM ---
            st.markdown("<br>**3. Structure de la clientèle (RFM)**", unsafe_allow_html=True)
            rfm_dist = df_brand['RFM_Name'].value_counts().reset_index()
            rfm_dist.columns = ['Segment', 'Volume']
            fig3 = px.pie(rfm_dist, names='Segment', values='Volume', hole=0.4, color_discrete_sequence=['#000000', '#E50000', '#666666', '#CCCCCC'])
            fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig3, use_container_width=True)
    # ==========================================
    # ONGLET 5 : STRATÉGIE D'ACQUISITION (SOLDES & SAISONNALITÉ)
    # ==========================================
    with tab5:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Analyse de l'Impact des Soldes sur l'Acquisition**")
        st.write("Ce module identifie les pics de recrutement et analyse si les périodes promotionnelles (Soldes, Black Friday) sont les moteurs principaux des nouveaux clients.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Filtrage des nouveaux clients (Segment 4 - New 3M)
        df_new = df[df['RFM_Name'] == '4 - New 3M'].copy()
        
        if not df_new.empty:
            # Conversion de la date et création d'une colonne Mois
            df_new['transactionDate'] = pd.to_datetime(df_new['transactionDate'])
            df_new['Mois'] = df_new['transactionDate'].dt.strftime('%m - %B')
            
            # --- GRAPH 1 : SAISONNALITÉ DE L'ACQUISITION ---
            st.markdown("**1. Courbe de Recrutement Annuelle**")
            acq_temporelle = df_new.groupby('Mois')['anonymized_card_code'].nunique().reset_index()
            acq_temporelle.columns = ['Mois', 'Nouveaux Clients']
            acq_temporelle = acq_temporelle.sort_values('Mois')

            fig_time = px.line(acq_temporelle, x='Mois', y='Nouveaux Clients', markers=True,
                              title="Volume de nouveaux clients recrutés par mois",
                              color_discrete_sequence=['#E50000'])
            
            # On épaissit la ligne pour qu'elle reste bien visible par-dessus les blocs
            fig_time.update_traces(line=dict(width=4), marker=dict(size=10))
            
            # --- AJOUT DES 5 PÉRIODES CLÉS (Plus opaques et colorées) ---
            fig_time.add_vrect(x0="01 - January", x1="02 - February", fillcolor="#CCCCCC", opacity=0.4, line_width=0, annotation_text="❄️ Soldes Hiver", annotation_position="top left")
            fig_time.add_vrect(x0="05 - May", x1="05 - May", fillcolor="#FFB6C1", opacity=0.4, line_width=0, annotation_text="🌸 Fête des Mères", annotation_position="top left")
            fig_time.add_vrect(x0="06 - June", x1="07 - July", fillcolor="#CCCCCC", opacity=0.4, line_width=0, annotation_text="☀️ Soldes Été", annotation_position="top left")
            fig_time.add_vrect(x0="11 - November", x1="11 - November", fillcolor="#000000", opacity=0.25, line_width=0, annotation_text="🖤 Black Friday", annotation_position="top left")
            fig_time.add_vrect(x0="12 - December", x1="12 - December", fillcolor="#E50000", opacity=0.25, line_width=0, annotation_text="🎄 Noël", annotation_position="top left")
            
            # On force toutes les annotations en noir, un peu plus grosses
            fig_time.update_annotations(font=dict(size=13, color="#000000", family="Arial"))
            
            fig_time.update_layout(plot_bgcolor='white', yaxis_title="Nb de nouveaux clients", xaxis_title="")
            st.plotly_chart(fig_time, use_container_width=True)

            # --- ANALYSE DES MARQUES / CATÉGORIES ---
            col_marques, col_cat = st.columns(2)
            
            with col_marques:
                st.markdown("**2. Top 5 Gateway Brands (Recrutement)**")
                gateway = df_new.groupby('brand')['anonymized_card_code'].nunique().sort_values(ascending=False).head(5).reset_index()
                fig_gate = px.bar(gateway, x='anonymized_card_code', y='brand', orientation='h', color_discrete_sequence=['#000000'])
                fig_gate.update_layout(plot_bgcolor='white', xaxis_title="Clients recrutés", yaxis_title="")
                st.plotly_chart(fig_gate, use_container_width=True)

            with col_cat:
                st.markdown("**3. Catégories d'Entrée**")
                entry = df_new['Axe_Desc'].value_counts().reset_index().head(5)
                fig_pie = px.pie(entry, names='Axe_Desc', values='count', hole=0.5, color_discrete_sequence=['#E50000', '#000000', '#666666'])
                st.plotly_chart(fig_pie, use_container_width=True)

            st.info("💡 **Insight Actionnable :** Si les pics coïncident avec les soldes (Janvier/Juin), Sephora doit activer un programme de fidélisation agressif à M+1 pour éviter que ces clients 'chasseurs de prix' ne deviennent inactifs.")

    # ==========================================
    # ONGLET 6 : GÉNÉRATEUR DE PERSONAS (VERSION TABLEAU)
    # ==========================================
    with tab6:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Générateur de Personas Data-Driven**")
        st.write("Analyse comportementale et psychographique des segments clients sous forme de fiche technique.")
        
        liste_segments = sorted(df['RFM_Name'].dropna().unique().tolist())
        selected_persona = st.selectbox("Sélectionnez le profil à analyser :", liste_segments)
        st.markdown("</div>", unsafe_allow_html=True)

        if selected_persona:
            df_persona = df[df['RFM_Name'] == selected_persona]
            
            # --- CALCULS ---
            nb_clients = df_persona['anonymized_card_code'].nunique()
            ca_total = df_persona['salesVatEUR'].sum()
            tickets_uniques = df_persona['anonymized_Ticket_ID'].nunique()
            panier_moyen = ca_total / tickets_uniques if tickets_uniques > 0 else 0
            freq_moyenne = tickets_uniques / nb_clients if nb_clients > 0 else 0
            
            top_brands = ", ".join(df_persona['brand'].value_counts().head(3).index.tolist())
            top_category = df_persona['Axe_Desc'].value_counts().idxmax() if not df_persona.empty else "N/A"
            top_store = df_persona['store_city'].value_counts().idxmax() if not df_persona.empty else "N/A"
            

            # --- DÉFINITION DU NARRATIF ---
            if "VIP" in selected_persona:
                nom = "L'Ambassadrice Beauté 👑"
                strategie = "Rétention exclusive & Avant-premières"
            elif "Good" in selected_persona:
                nom = "L'Acheteuse Régulière 🛍️"
                strategie = "Cross-selling & Augmentation du panier"
            elif "Opportunist" in selected_persona:
                nom = "La Chasseuse de Bons Plans 🎯"
                strategie = "Réactivation hors-soldes & Alertes promo"
            else:
                nom = "La Recrue en Découverte 🌱"
                strategie = "Onboarding & Conversion 2ème achat"

            # --- CRÉATION DU TABLEAU DE SYNTHÈSE ---
            st.markdown(f"### 📋 Fiche Persona : {nom}")
            
            persona_table = {
                "Dimension": [
                    "👤 Nom du Profil",
                    "📊 Volume du Segment",
                    "💰 Valeur Panier Moyen",
                    "🔄 Fréquence d'Achat",
                    "🏷️ Catégorie Favorite",
                    "⭐ Top 3 Marques",
                    "📍 Lieu de prédilection",
                    "🚀 Stratégie CRM recommandée"
                ],
                "Données Analytiques": [
                    nom,
                    f"{nb_clients:,} clients",
                    f"{panier_moyen:.2f} €",
                    f"{freq_moyenne:.1f} fois / an",
                    top_category,
                    top_brands,
                    top_store,
                    strategie
                ]
            }
            
            # Affichage du tableau
            st.dataframe(pd.DataFrame(persona_table), use_container_width=True, hide_index=True)

            # --- PETIT GRAPHIQUE DE RÉPARTITION (En dessous pour le support visuel) ---
            st.markdown("<br>**Détail des dépenses par Axe Produit**", unsafe_allow_html=True)
            dist_budget = df_persona.groupby('Axe_Desc')['salesVatEUR'].sum().reset_index()
            fig_p = px.bar(dist_budget, x='salesVatEUR', y='Axe_Desc', orientation='h',
                           color_discrete_sequence=['#000000'],
                           labels={'salesVatEUR': 'Dépenses Totales (€)', 'Axe_Desc': ''})
            fig_p.update_layout(plot_bgcolor='white', height=300, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_p, use_container_width=True)
    # ==========================================
    # ONGLET 7 : ANALYSE GÉNÉRATIONNELLE (L'approche du Groupe 15, en mieux)
    # ==========================================
    with tab7:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Analyse des Comportements d'Achat par Génération**")
        st.write("Ce module stratifie les habitudes de consommation par tranche d'âge, tout en gardant à l'esprit que les données démographiques sont déclaratives et partiellement manquantes.")
        st.markdown("</div>", unsafe_allow_html=True)

        if 'age' in df.columns or 'Age' in df.columns:
            # Sécurisation du nom de la colonne
            age_col = 'age' if 'age' in df.columns else 'Age'
            
            with st.spinner("Calcul des agrégations inter-générationnelles..."):
                # 1. Création des tranches d'âge (Mapping)
                def get_generation(age_val):
                    if pd.isna(age_val) or age_val < 10 or age_val > 100: return 'Inconnu / Erreur'
                    elif age_val <= 14: return 'Gen A'
                    elif age_val <= 27: return 'Gen Z'
                    elif age_val <= 43: return 'Millennials'
                    elif age_val <= 59: return 'Gen X'
                    else: return 'Baby Boomers'

                # Copie locale pour ne pas polluer le df global
                df_gen = df.copy()
                df_gen['Generation'] = df_gen[age_col].apply(get_generation)
                
                # On exclut les inconnus pour le tableau d'analyse pur
                df_clean_gen = df_gen[df_gen['Generation'] != 'Inconnu / Erreur']

                # 2. Calcul de la "Global Baseline" (Moyenne Globale)
                tickets_globaux = df.groupby('anonymized_Ticket_ID').agg(
                    nb_brands=('brand', 'nunique'),
                    spend=('salesVatEUR', 'sum')
                )
                global_avg_brands = tickets_globaux['nb_brands'].mean()
                global_multi_brand_rate = (tickets_globaux['nb_brands'] > 1).mean() * 100
                global_avg_spend = tickets_globaux['spend'].mean()

                # 3. Calculs par Génération
                gen_stats = []
                for gen in ['Gen A', 'Gen Z', 'Millennials', 'Gen X', 'Baby Boomers']:
                    df_g = df_clean_gen[df_clean_gen['Generation'] == gen]
                    if not df_g.empty:
                        tickets_g = df_g.groupby('anonymized_Ticket_ID').agg(
                            nb_brands=('brand', 'nunique'),
                            spend=('salesVatEUR', 'sum')
                        )
                        gen_stats.append({
                            'Génération': gen,
                            'Volume Clients': df_g['anonymized_card_code'].nunique(),
                            'Marques / Panier': tickets_g['nb_brands'].mean(),
                            'Taux Multi-Marques (%)': (tickets_g['nb_brands'] > 1).mean() * 100,
                            'Panier Moyen (€)': tickets_g['spend'].mean()
                        })

                # 4. Formatage du Tableau de Synthèse
                df_results = pd.DataFrame(gen_stats)
                
                # Ajout de la ligne Baseline
                baseline_row = pd.DataFrame([{
                    'Génération': '🌍 GLOBAL BASELINE',
                    'Volume Clients': df['anonymized_card_code'].nunique(),
                    'Marques / Panier': global_avg_brands,
                    'Taux Multi-Marques (%)': global_multi_brand_rate,
                    'Panier Moyen (€)': global_avg_spend
                }])
                
                df_results = pd.concat([baseline_row, df_results], ignore_index=True)
                
                # Arrondir pour faire propre
                df_results['Marques / Panier'] = df_results['Marques / Panier'].round(2)
                df_results['Taux Multi-Marques (%)'] = df_results['Taux Multi-Marques (%)'].round(1)
                df_results['Panier Moyen (€)'] = df_results['Panier Moyen (€)'].round(2)

                # Affichage
                st.markdown("**1. Comparaison Stratifiée (Benchmark vs Baseline)**")
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # 5. Graphique Visuel de Dépense
                st.markdown("**2. Écart au Panier Moyen Global**")
                df_plot = df_results[df_results['Génération'] != '🌍 GLOBAL BASELINE'].copy()
                df_plot['Écart Baseline (€)'] = df_plot['Panier Moyen (€)'] - global_avg_spend
                df_plot['Couleur'] = df_plot['Écart Baseline (€)'].apply(lambda x: '#118D57' if x > 0 else '#E50000')
                
                fig_gen = px.bar(df_plot, x='Génération', y='Écart Baseline (€)', 
                                 text_auto='.2f', color='Couleur', color_discrete_map='identity',
                                 title="Sur/Sous-performance du Panier Moyen par rapport au Global")
                fig_gen.update_layout(plot_bgcolor='white', margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig_gen, use_container_width=True)

                # 6. Le Warning de "Data Mastery" pour le Jury
                st.warning("""
                **⚠️ Alerte Méthodologique (Data Mastery) :** Bien que ces insights soient intéressants pour le marketing de contenu, **nous déconseillons d'utiliser l'âge comme variable principale pour le moteur de recommandation**. Nos audits révèlent que les données démographiques sont manquantes ou peu fiables pour près de 30% de la base. C'est pourquoi notre algorithme principal (Onglet Moteur d'Affinité) se base sur le RFM et le comportement d'achat réel, qui sont fiables à 100%.
                """)
        else:
            st.error("La colonne 'age' est introuvable dans le dataset. Vérifiez le nom de la colonne démographique.")




    # ==========================================
    # ONGLET 8 : PRÉDICTION DE LA LTV (LIFETIME VALUE)
    # ==========================================
    with tab8:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Modèle Prédictif de la Valeur Client (LTV à 3 ans)**")
        st.write("Cet algorithme estime le Chiffre d'Affaires qu'un client va générer sur les 3 prochaines années en fonction de son comportement actuel (Panier Moyen x Fréquence x Marge).")
        st.markdown("</div>", unsafe_allow_html=True)

        col_param, col_calc = st.columns([1, 1.5], gap="large")
        
        with col_param:
            st.markdown("**1. Choix du Profil Client**")
            liste_segments = sorted(df['RFM_Name'].dropna().unique().tolist())
            ltv_segment = st.selectbox("Sélectionnez le segment pour prédire sa valeur future :", liste_segments, key="ltv_box")
            
            st.markdown("**2. Hypothèse Financière**")
            marge_brute = st.slider("Marge Brute estimée de Sephora (%) :", min_value=30, max_value=80, value=50, step=5)
            
        with col_calc:
            df_ltv = df[df['RFM_Name'] == ltv_segment]
            if not df_ltv.empty:
                # Calculs réels basés sur l'historique du segment
                nb_clients = df_ltv['anonymized_card_code'].nunique()
                ca_total = df_ltv['salesVatEUR'].sum()
                nb_tickets = df_ltv['anonymized_Ticket_ID'].nunique()
                
                # Variables du modèle LTV
                aov = ca_total / nb_tickets if nb_tickets > 0 else 0  # Average Order Value
                f_an = nb_tickets / nb_clients if nb_clients > 0 else 0 # Purchase Frequency
                lifespan = 3 # Estimation sur 3 ans
                
                # Formule de la LTV (Valeur brute générée)
                ltv_revenue = aov * f_an * lifespan
                # Formule de la LTV (Profit net)
                ltv_profit = ltv_revenue * (marge_brute / 100)
                
                st.markdown(f"### Projection pour le segment : {ltv_segment}")
                
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='sephora-card' style='border-left: 4px solid #000;'><div class='metric-title'>Valeur Future Brut (CA sur 3 ans)</div><div class='metric-value'>{ltv_revenue:.2f} € / client</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='sephora-card' style='border-left: 4px solid #E50000;'><div class='metric-title'>Profit Net Estimé (LTV Réelle)</div><div class='metric-value'>{ltv_profit:.2f} € / client</div></div>", unsafe_allow_html=True)
                
                st.info(f"💡 **Insight :** Sephora peut se permettre de dépenser jusqu'à **{ltv_profit * 0.2:.2f} €** en coût d'acquisition (CAC) ou en cadeaux CRM pour recruter/retenir un client de ce segment, tout en restant hautement rentable.")
    
    # ==========================================
    # ONGLET 9 : SIMULATEUR DE R.O.I (BUSINESS CASE)
    # ==========================================
    with tab9:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Simulateur d'Impact Financier (Business Case)**")
        st.write("Estimez les gains financiers (Uplift) générés par l'implémentation de nos recommandations IA.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Calcul des bases de référence
        df_vip = df[df['RFM_Name'] == '1 - VIP']
        df_new = df[df['RFM_Name'] == '4 - New 3M']
        
        ca_actuel = df['salesVatEUR'].sum()
        panier_vip = df_vip['salesVatEUR'].sum() / df_vip['anonymized_card_code'].nunique() if not df_vip.empty else 200
        panier_new = df_new['salesVatEUR'].sum() / df_new['anonymized_card_code'].nunique() if not df_new.empty else 50
        
        col_sliders, col_kpi = st.columns([1, 1.5], gap="large")
        
        with col_sliders:
            st.markdown("**1. Hypothèses de Performance (Leviers CRM)**")
            st.info("Ajustez les curseurs selon les objectifs fixés aux équipes :")
            
            conversion_rate = st.slider(
                "Taux de conversion (Nouveau ➡️ VIP) :", 
                min_value=0.0, max_value=15.0, value=2.0, step=0.5,
                help="Pourcentage de Nouveaux convertis grâce à la bonne Gateway Brand."
            )
            
            anti_churn_rate = st.slider(
                "Sauvetage Anti-Churn (VIP) :", 
                min_value=0.0, max_value=10.0, value=1.5, step=0.5,
                help="VIP inactifs réactivés grâce au moteur d'Affinité."
            )

        with col_kpi:
            st.markdown("**2. Projection de Croissance Annuelle**")
            
            nb_new = df_new['anonymized_card_code'].nunique() if not df_new.empty else 0
            nb_vip_churners = int(df_vip['anonymized_card_code'].nunique() * 0.15)
            
            gain_conversion = (nb_new * (conversion_rate / 100)) * (panier_vip - panier_new)
            gain_retention = (nb_vip_churners * (anti_churn_rate / 100)) * panier_vip
            
            uplift_total = gain_conversion + gain_retention
            croissance = (uplift_total / ca_actuel) * 100 if ca_actuel > 0 else 0

            st.markdown(f"""
            <div class="sephora-card" style="border: 2px solid #118D57; background-color: #F8FFF9 !important; text-align: center; padding: 30px;">
                <h3 style="color: #118D57 !important; margin:0;">💰 Gain Potentiel (Uplift)</h3>
                <h1 style="color: #118D57 !important; font-size: 48px; margin: 10px 0;">+ {uplift_total:,.0f} €</h1>
                <p style="color: #666; font-size: 16px; margin:0;">Soit une croissance additionnelle de <b>+{croissance:.2f}%</b> sur l'échantillon analysé.</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='sephora-card' style='border-left: 4px solid #000;'><div class='metric-title'>Gain via Acquisition</div><div class='metric-value'>+ {gain_conversion:,.0f} €</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='sephora-card' style='border-left: 4px solid #E50000;'><div class='metric-title'>Gain via Rétention</div><div class='metric-value'>+ {gain_retention:,.0f} €</div></div>", unsafe_allow_html=True)







    # ==========================================
    # ONGLET 10 : AUDIT & DICTIONNAIRE DATA
    # ==========================================
    with tab10:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Diagnostic de Santé du Dataset Importé**")
        st.write("Cet onglet analyse en temps réel la qualité des données de l'échantillon fourni.")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 1. APERÇU GLOBAL (Métriques de base) ---
        st.markdown("**1. Volumétrie Globale**")
        nb_lignes = df.shape[0]
        nb_colonnes = df.shape[1]
        nb_clients_uniques = df['anonymized_card_code'].nunique() if 'anonymized_card_code' in df.columns else 0
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='sephora-card' style='border-left: 4px solid #000;'><div class='metric-title'>Total Lignes (Transactions)</div><div class='metric-value'>{nb_lignes:,}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='sephora-card' style='border-left: 4px solid #000;'><div class='metric-title'>Total Colonnes (Variables)</div><div class='metric-value'>{nb_colonnes}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='sephora-card' style='border-left: 4px solid #E50000;'><div class='metric-title'>Clients Uniques</div><div class='metric-value'>{nb_clients_uniques:,}</div></div>", unsafe_allow_html=True)

        # --- 2. AUDIT DES VALEURS MANQUANTES (Calculé en live) ---
        col_gauche, col_droite = st.columns([1, 1], gap="large")
        
        with col_gauche:
            st.markdown("**2. Taux de Valeurs Manquantes (Missingness)**")
            # Calcul des valeurs manquantes
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Colonne', 'Valeurs Vides']
            missing_data['% Manquant'] = (missing_data['Valeurs Vides'] / nb_lignes) * 100
            
            # On ne garde que les colonnes qui ont des trous, triées par ordre décroissant
            missing_data = missing_data[missing_data['Valeurs Vides'] > 0].sort_values(by='% Manquant', ascending=False)
            
            if not missing_data.empty:
                # Formatage pour un bel affichage
                missing_data['% Manquant'] = missing_data['% Manquant'].map('{:.1f}%'.format)
                st.dataframe(missing_data, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Aucune valeur manquante détectée dans ce dataset !")

        # --- 3. DICTIONNAIRE DES DONNÉES (Pour le Jury) ---
        with col_droite:
            st.markdown("**3. Dictionnaire des Variables Clés**")
            
            dico_data = {
                "Nom de la Colonne": [
                    "anonymized_card_code", 
                    "RFM_Segment_ID", 
                    "salesVatEUR", 
                    "anonymized_Ticket_ID", 
                    "brand", 
                    "Axe_Desc",
                    "age / gender"
                ],
                "Définition (Métier Sephora)": [
                    "Identifiant unique du client (Haché pour RGPD).",
                    "Score de fidélité (1 = VIP, 2 = Bon, 3 = Opportuniste, 4 = Nouveau).",
                    "Chiffre d'affaires généré par la ligne d'achat (en Euros).",
                    "Identifiant du panier de caisse (Regroupe les produits achetés ensemble).",
                    "Marque du produit acheté.",
                    "Catégorie globale (Make Up, Skincare, Fragrance, Haircare).",
                    "Données démographiques (Attention : souvent manquantes ou erronées)."
                ]
            }
            df_dico = pd.DataFrame(dico_data)
            st.dataframe(df_dico, use_container_width=True, hide_index=True)

        # --- 4. ALERTE QUALITÉ (Bonus "Data Rigor") ---
        st.markdown("**4. Avertissements sur la Qualité des Données (Data Quality Warnings)**")
        st.warning("""
        **Observations issues de notre Analyse Exploratoire (Deliverable 1) :**
        - **Cold Start Problem :** Plus de 74% des données liées au *premier achat* sont manquantes. Nous avons pivoté l'analyse sur l'historique récent plutôt que sur l'acquisition originelle.
        - **Biais Démographique :** Près de 28% de données manquantes sur l'âge et le sexe. C'est pourquoi notre algorithme de recommandation (Affinité de Marque) se base exclusivement sur le comportement d'achat (RFM) et non sur le profil sociodémographique.
        """)
    