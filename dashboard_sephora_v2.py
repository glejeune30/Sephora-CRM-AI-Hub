import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION ET DESIGN ---
st.set_page_config(page_title="Sephora AI Hub", layout="wide")

# --- 2. IMPORTATION AUTOMATIQUE DES DONNÉES ---
# @st.cache_data permet de garder le fichier en mémoire pour que l'app soit super rapide
@st.cache_data 
def load_data():
    # Remplace le nom du fichier par le nom EXACT de ton fichier CSV sur GitHub
    df = pd.read_csv("BDD#7_Database_Albert_School_Sephora.csv") 
    
    # Nettoyage automatique
    if 'Axe_Desc' in df.columns:
        df['Axe_Desc'] = df['Axe_Desc'].replace({'MAEK UP': 'MAKE UP'})
    rfm_mapping = {1: "1 - VIP", 2: "2 - Good Customer", 3: "3 - Opportunist", 4: "4 - New 3M"}
    if 'RFM_Segment_ID' in df.columns:
        df['RFM_Name'] = df['RFM_Segment_ID'].map(rfm_mapping)
        
    return df

with st.spinner("Chargement de la base de données Sephora..."):
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Erreur : Le fichier CSV est introuvable. Vérifiez qu'il est bien sur GitHub.")
        st.stop() # Arrête l'application si le fichier n'est pas là

# Le Header simple
st.markdown("<h2 style='margin-top:0;'>💄 Sephora AI Hub</h2>", unsafe_allow_html=True)
st.markdown("---")

# === CREATING THE MAIN MENU (10 TABS) ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🏠 Home", 
        "🎯 Affinity Engine", 
        "🧬 Segmentation Lab", 
        "📊 Brand Scorecard",
        "🚀 Acquisition",
        "👤 Personas",
        "👨‍👩‍👧‍👦 Generational Analysis", 
        "🔮 LTV Prediction",
        "💸 R.O.I Simulator",
        "🔎 Data Audit"
    ])
    
    # ==========================================
    # TAB 1 : EXECUTIVE COCKPIT (HOME PAGE)
    # ==========================================
with tab1:
        st.markdown("<h3 style='text-align: center;'>Welcome to the Sephora CRM Hub 🚀</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 30px;'>Artificial Intelligence driving Loyalty and Growth.</p>", unsafe_allow_html=True)
        
        # Calculating Super-KPIs
        ca_total = df['salesVatEUR'].sum()
        clients_actifs = df['anonymized_card_code'].nunique()
        tickets_totaux = df['anonymized_Ticket_ID'].nunique()
        panier_moyen_global = ca_total / tickets_totaux if tickets_totaux > 0 else 0
        
        # Displaying main metrics
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="sephora-card" style="border-top: 4px solid #000; text-align: center;">
            <div class="metric-title">Analyzed Revenue</div>
            <div class="metric-value" style="font-size: 32px;">{ca_total:,.0f} €</div>
        </div>""", unsafe_allow_html=True)
        
        c2.markdown(f"""
        <div class="sephora-card" style="border-top: 4px solid #000; text-align: center;">
            <div class="metric-title">Active Customer Base</div>
            <div class="metric-value" style="font-size: 32px;">{clients_actifs:,}</div>
        </div>""", unsafe_allow_html=True)
        
        c3.markdown(f"""
        <div class="sephora-card" style="border-top: 4px solid #E50000; text-align: center;">
            <div class="metric-title">Global Average Basket</div>
            <div class="metric-value" style="font-size: 32px;">{panier_moyen_global:.2f} €</div>
        </div>""", unsafe_allow_html=True)
        
        # Mission summary
        st.markdown("""
        <div class="sephora-card">
            <h4>🎯 Dashboard Mission</h4>
            <p>This interactive cockpit answers <b>Use Case n°3: "Brand Affinity & Cold Start"</b>. Our goal is to go beyond simple Market Basket analysis by integrating the customer's maturity level (RFM) to personalize recommendations.</p>
            <ul>
                <li><b>For CRM:</b> Use the <i>Affinity Engine</i> to target the right campaigns to the right people.</li>
                <li><b>For Marketing:</b> Use the <i>Personas</i> and the <i>Segmentation Lab</i> to understand purchasing psychology.</li>
                <li><b>For Management:</b> Use the <i>ROI Simulator</i> to quantify the impact of our algorithmic models.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # --- ADDING THE QR CODE FOR THE JURY ---
        st.markdown("---")
        col_qr_text, col_qr_img = st.columns([3, 1])
        
        with col_qr_text:
            st.markdown("### 📱 Test the app live")
            st.write("Scan this QR Code with your smartphone or tablet to access the Sephora AI Hub. No installation or file import is required.")
        
        with col_qr_img:
            # REPLACE THIS URL WITH YOUR ACTUAL STREAMLIT URL
            app_url = "https://sephora-crm-ai-hub.streamlit.app/" 
            
            # Generating the QR Code via a free API
            qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={app_url}"
            st.image(qr_api_url, width=150)

# ==========================================
    # TAB 2 : AFFINITY ENGINE (The Return of the Graph)
    # ==========================================
with tab2:
        # --- THE STRATEGIC MATRIX (Corrected and Readable) ---
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**1. Brand Strategic Positioning Matrix**")
        st.write("Mapping of the Sephora catalog based on average buyer loyalty and generated value. Only the top 15 largest brands are displayed in text for readability (hover over other bubbles for details).")
        
        with st.spinner("Generating brand matrix..."):
            if 'RFM_Segment_ID' in df.columns:
                # 1. Calculations
                brand_matrix = df.groupby('brand').agg(
                    avg_rfm=('RFM_Segment_ID', 'mean'),
                    avg_spend=('salesVatEUR', 'mean'),
                    clients_uniques=('anonymized_card_code', 'nunique')
                ).reset_index()
                
                # 2. Noise filter (Removing micro-brands that skew the axis)
                brand_matrix = brand_matrix[brand_matrix['clients_uniques'] > 50]
                
                # 3. Smart labeling (Only Top 15 as visible text)
                top_brands = brand_matrix.nlargest(15, 'clients_uniques')['brand'].tolist()
                brand_matrix['label'] = brand_matrix['brand'].apply(lambda x: x if x in top_brands else "")
                
                # 4. Plotly Graph
                fig_matrix = px.scatter(
                    brand_matrix, x='avg_rfm', y='avg_spend', 
                    size='clients_uniques', color='avg_spend',
                    hover_name='brand', text='label',
                    color_continuous_scale=['#000000', '#E50000'],
                    labels={'avg_rfm': "<- VIPs (1) | Average Loyalty | New (4) ->", 
                            'avg_spend': "Average Spend (€)"},
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
            st.markdown("**1. Define Target Audience**")
            liste_segments = sorted(df['RFM_Name'].dropna().unique().tolist())
            liste_categories = sorted(df['Axe_Desc'].dropna().unique().tolist())
            
            segment = st.selectbox("Target RFM Segment:", liste_segments)
            categorie = st.selectbox("Category of Interest:", liste_categories)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_resultats:
            st.markdown("**2. Affinity Analysis (Lift Algorithm)**")
            
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
                            <div class="metric-title" style="color: {title_color} !important;">Choice n°{rank}</div>
                            <div class="metric-value" style="color: {text_color} !important;">{brand_name}</div>
                            <div style="color: {text_color} !important;">Affinity (Lift): <span class="badge-green">x{lift_val:.2f}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(top_brands) > 0: display_card(c1, 1, top_brands.index[0], top_brands.iloc[0]['Lift'], top_brands.iloc[0]['Segment'])
                    if len(top_brands) > 1: display_card(c2, 2, top_brands.index[1], top_brands.iloc[1]['Lift'], top_brands.iloc[1]['Segment'])
                    if len(top_brands) > 2: display_card(c3, 3, top_brands.index[2], top_brands.iloc[2]['Lift'], top_brands.iloc[2]['Segment'])
                    
                    # THE PLOTLY GRAPH (Clear and readable)
                    plot_data = top_brands.head(5).reset_index()
                    if 'brand' in plot_data.columns: plot_data = plot_data.rename(columns={'brand': 'Brand'})
                    elif 'index' in plot_data.columns: plot_data = plot_data.rename(columns={'index': 'Brand'})

                    fig = px.bar(plot_data, x='Brand', y=['Segment', 'Global'], barmode='group',
                                 title=f"Analytical Proof: Over-performance on segment '{segment}'",
                                 labels={'value': 'Market Share', 'variable': 'Cohort (Segment vs Global)'},
                                 color_discrete_sequence=['#E50000', '#000000'])
                    fig.update_layout(plot_bgcolor='white', margin=dict(t=40, l=0, r=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for this segment.")

    # ==========================================
    # TAB 3 : SEGMENTATION LAB (K-Means Explained)
    # ==========================================
with tab3:
        col_k, col_plot = st.columns([1, 2.5], gap="large")
        
        with col_k:
            st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
            st.markdown("**AI Configuration**")
            st.write("The algorithm groups customers based on their actual behavior, without prior assumptions.")
            k_clusters = st.slider("Number of profiles to generate (K):", min_value=2, max_value=5, value=3)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_plot:
            with st.spinner("Running K-Means analysis..."):
                # 1. Preparation (Aggregation by customer)
                customer_df = df.groupby('anonymized_card_code').agg(
                    Depense_Totale=('salesVatEUR', 'sum'),
                    Frequence_Achat=('anonymized_Ticket_ID', 'nunique')
                ).reset_index()
                
                customer_df = customer_df[(customer_df['Depense_Totale'] > 0) & (customer_df['Depense_Totale'] < 2000)]
                
                # 2. Machine Learning Model
                features = customer_df[['Depense_Totale', 'Frequence_Achat']]
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                customer_df['Cluster_ID'] = kmeans.fit_predict(features_scaled)
                customer_df['Cluster_Name'] = "Profile " + customer_df['Cluster_ID'].astype(str)
                
                # 3. DECODING (Making it understandable)
                st.markdown("**1. Decoding the profiles identified by AI (Centroids)**")
                st.write("Here are the average characteristics of the groups the algorithm just created:")
                
                summary_df = customer_df.groupby('Cluster_Name').agg(
                    Number_of_Customers=('anonymized_card_code', 'count'),
                    Global_Average_Basket=('Depense_Totale', 'mean'),
                    Average_Frequency=('Frequence_Achat', 'mean')
                ).round(1).reset_index()
                
                # Displaying a nice explanatory table
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # 4. VISUAL GRAPH
                st.markdown("**2. Distribution Visualization**")
                fig2 = px.scatter(customer_df, x='Frequence_Achat', y='Depense_Totale', color='Cluster_Name',
                                 labels={'Frequence_Achat': 'Frequency (Number of tickets)', 'Depense_Totale': 'Total Cumulative Spend (€)'},
                                 color_discrete_sequence=px.colors.qualitative.Bold,
                                 opacity=0.7)
                fig2.update_layout(plot_bgcolor='white', margin=dict(t=10, l=0, r=0, b=0))
                st.plotly_chart(fig2, use_container_width=True)    
# ==========================================
    # TAB 4 : BRAND SCORECARD (Scorecard & Correlation)
    # ==========================================
with tab4:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**1. Select a brand to analyze:**")
        liste_marques = sorted(df['brand'].dropna().unique().tolist())
        selected_brand = st.selectbox("", liste_marques, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        if selected_brand:
            df_brand = df[df['brand'] == selected_brand]
            
            # --- 1. BASIC CALCULATIONS ---
            ca_total = df_brand['salesVatEUR'].sum()
            tickets_uniques = df_brand['anonymized_Ticket_ID'].unique()
            nb_tickets = len(tickets_uniques)
            panier_moyen = ca_total / nb_tickets if nb_tickets > 0 else 0
            
            # --- 2. TOP CALCULATIONS (Location & Category) ---
            top_store = df_brand.groupby('store_city')['salesVatEUR'].sum().idxmax() if not df_brand.empty else "N/A"
            top_category = df_brand.groupby('Axe_Desc')['salesVatEUR'].sum().idxmax() if not df_brand.empty else "N/A"
            
            # --- 3. CORRELATION ANALYSIS (BASKET) ---
            paniers_contenant_marque = df[df['anonymized_Ticket_ID'].isin(tickets_uniques)]
            autres_produits_panier = paniers_contenant_marque[paniers_contenant_marque['brand'] != selected_brand]
            
            if not autres_produits_panier.empty:
                corrélation_stats = autres_produits_panier['brand'].value_counts()
                top_companion_brand = corrélation_stats.idxmax()
                nb_co_occurrence = corrélation_stats.max()
                taux_attachement = (nb_co_occurrence / nb_tickets) * 100
            else:
                top_companion_brand = "None"
                taux_attachement = 0

            # --- 4. DISPLAY KPIS IN A CLEAN TABLE ---
            st.markdown("**2. Identity & Performance Scorecard**")
            
            scorecard_data = {
                "Key Indicator": [
                    "💰 Total Revenue",
                    "🛍️ Average Basket",
                    "📍 Top Store (Revenue)",
                    "🏷️ Top Category",
                    "🔥 Most bought with (Correlation)"
                ],
                "Analytical Result": [
                    f"{ca_total:,.0f} €".replace(',', ' '),
                    f"{panier_moyen:.2f} €",
                    top_store,
                    top_category,
                    f"{top_companion_brand} (Present in {taux_attachement:.1f}% of baskets)"
                ]
            }
            
            df_scorecard = pd.DataFrame(scorecard_data)
            st.dataframe(df_scorecard, use_container_width=True, hide_index=True)
            
            # --- 5. RFM GRAPH ---
            st.markdown("<br>**3. Customer Structure (RFM)**", unsafe_allow_html=True)
            rfm_dist = df_brand['RFM_Name'].value_counts().reset_index()
            rfm_dist.columns = ['Segment', 'Volume']
            fig3 = px.pie(rfm_dist, names='Segment', values='Volume', hole=0.4, color_discrete_sequence=['#000000', '#E50000', '#666666', '#CCCCCC'])
            fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig3, use_container_width=True)

    # ==========================================
    # TAB 5 : ACQUISITION STRATEGY (SALES & SEASONALITY)
    # ==========================================
with tab5:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Analysis of Sales Impact on Acquisition**")
        st.write("This module identifies recruitment peaks and analyzes whether promotional periods (Sales, Black Friday) are the main drivers for new customers.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Filtering new customers (Segment 4 - New 3M)
        df_new = df[df['RFM_Name'] == '4 - New 3M'].copy()
        
        if not df_new.empty:
            # Date conversion and Month column creation
            df_new['transactionDate'] = pd.to_datetime(df_new['transactionDate'])
            df_new['Month'] = df_new['transactionDate'].dt.strftime('%m - %B')
            
            # --- GRAPH 1 : ACQUISITION SEASONALITY ---
            st.markdown("**1. Annual Recruitment Curve**")
            acq_temporelle = df_new.groupby('Month')['anonymized_card_code'].nunique().reset_index()
            acq_temporelle.columns = ['Month', 'New Customers']
            acq_temporelle = acq_temporelle.sort_values('Month')

            fig_time = px.line(acq_temporelle, x='Month', y='New Customers', markers=True,
                              title="Volume of new customers recruited per month",
                              color_discrete_sequence=['#E50000'])
            
            # Thickening the line so it remains visible over the blocks
            fig_time.update_traces(line=dict(width=4), marker=dict(size=10))
            
            # --- ADDING THE 5 KEY PERIODS (More opaque and colored) ---
            fig_time.add_vrect(x0="01 - January", x1="02 - February", fillcolor="#CCCCCC", opacity=0.4, line_width=0, annotation_text="❄️ Winter Sales", annotation_position="top left")
            fig_time.add_vrect(x0="05 - May", x1="05 - May", fillcolor="#FFB6C1", opacity=0.4, line_width=0, annotation_text="🌸 Mother's Day", annotation_position="top left")
            fig_time.add_vrect(x0="06 - June", x1="07 - July", fillcolor="#CCCCCC", opacity=0.4, line_width=0, annotation_text="☀️ Summer Sales", annotation_position="top left")
            fig_time.add_vrect(x0="11 - November", x1="11 - November", fillcolor="#000000", opacity=0.25, line_width=0, annotation_text="🖤 Black Friday", annotation_position="top left")
            fig_time.add_vrect(x0="12 - December", x1="12 - December", fillcolor="#E50000", opacity=0.25, line_width=0, annotation_text="🎄 Christmas", annotation_position="top left")
            
            # Forcing all annotations to black, slightly larger
            fig_time.update_annotations(font=dict(size=13, color="#000000", family="Arial"))
            
            fig_time.update_layout(plot_bgcolor='white', yaxis_title="Number of new customers", xaxis_title="")
            st.plotly_chart(fig_time, use_container_width=True)

            # --- BRAND / CATEGORY ANALYSIS ---
            col_marques, col_cat = st.columns(2)
            
            with col_marques:
                st.markdown("**2. Top 5 Gateway Brands (Recruitment)**")
                gateway = df_new.groupby('brand')['anonymized_card_code'].nunique().sort_values(ascending=False).head(5).reset_index()
                fig_gate = px.bar(gateway, x='anonymized_card_code', y='brand', orientation='h', color_discrete_sequence=['#000000'])
                fig_gate.update_layout(plot_bgcolor='white', xaxis_title="Recruited customers", yaxis_title="")
                st.plotly_chart(fig_gate, use_container_width=True)

            with col_cat:
                st.markdown("**3. Entry Categories**")
                entry = df_new['Axe_Desc'].value_counts().reset_index().head(5)
                fig_pie = px.pie(entry, names='Axe_Desc', values='count', hole=0.5, color_discrete_sequence=['#E50000', '#000000', '#666666'])
                st.plotly_chart(fig_pie, use_container_width=True)

            st.info("💡 **Actionable Insight:** If peaks coincide with sales (January/June), Sephora must activate an aggressive loyalty program at M+1 to prevent these 'price hunter' customers from becoming inactive.")

# ==========================================
    # TAB 6 : PERSONA GENERATOR (TABLE VERSION)
    # ==========================================
with tab6:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Data-Driven Persona Generator**")
        st.write("Behavioral and psychographic analysis of customer segments in a scorecard format.")
        
        liste_segments = sorted(df['RFM_Name'].dropna().unique().tolist())
        selected_persona = st.selectbox("Select the profile to analyze:", liste_segments)
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
                nom = "The Beauty Ambassador 👑"
                strategie = "Exclusive retention & Early access"
            elif "Good" in selected_persona:
                nom = "The Regular Shopper 🛍️"
                strategie = "Cross-selling & Basket size increase"
            elif "Opportunist" in selected_persona:
                nom = "The Bargain Hunter 🎯"
                strategie = "Off-sale reactivation & Promo alerts"
            else:
                nom = "The New Explorer 🌱"
                strategie = "Onboarding & 2nd purchase conversion"

            # --- CRÉATION DU TABLEAU DE SYNTHÈSE ---
            st.markdown(f"### 📋 Persona Card: {nom}")
            
            persona_table = {
                "Dimension": [
                    "👤 Profile Name",
                    "📊 Segment Volume",
                    "💰 Average Basket Value",
                    "🔄 Purchase Frequency",
                    "🏷️ Favorite Category",
                    "⭐ Top 3 Brands",
                    "📍 Preferred Location",
                    "🚀 Recommended CRM Strategy"
                ],
                "Analytical Data": [
                    nom,
                    f"{nb_clients:,} customers",
                    f"{panier_moyen:.2f} €",
                    f"{freq_moyenne:.1f} times / year",
                    top_category,
                    top_brands,
                    top_store,
                    strategie
                ]
            }
            
            # Affichage du tableau
            st.dataframe(pd.DataFrame(persona_table), use_container_width=True, hide_index=True)

            # --- PETIT GRAPHIQUE DE RÉPARTITION ---
            st.markdown("<br>**Spend breakdown by Product Category**", unsafe_allow_html=True)
            dist_budget = df_persona.groupby('Axe_Desc')['salesVatEUR'].sum().reset_index()
            fig_p = px.bar(dist_budget, x='salesVatEUR', y='Axe_Desc', orientation='h',
                           color_discrete_sequence=['#000000'],
                           labels={'salesVatEUR': 'Total Spend (€)', 'Axe_Desc': ''})
            fig_p.update_layout(plot_bgcolor='white', height=300, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_p, use_container_width=True)

    # ==========================================
    # TAB 7 : GENERATIONAL ANALYSIS (Group 15's approach, upgraded)
    # ==========================================
with tab7:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Purchasing Behavior Analysis by Generation**")
        st.write("This module stratifies consumption habits by age group, keeping in mind that demographic data is declarative and partially missing.")
        st.markdown("</div>", unsafe_allow_html=True)

        if 'age' in df.columns or 'Age' in df.columns:
            # Sécurisation du nom de la colonne
            age_col = 'age' if 'age' in df.columns else 'Age'
            
            with st.spinner("Calculating inter-generational aggregations..."):
                # 1. Création des tranches d'âge (Mapping)
                def get_generation(age_val):
                    if pd.isna(age_val) or age_val < 10 or age_val > 100: return 'Unknown / Error'
                    elif age_val <= 14: return 'Gen A'
                    elif age_val <= 27: return 'Gen Z'
                    elif age_val <= 43: return 'Millennials'
                    elif age_val <= 59: return 'Gen X'
                    else: return 'Baby Boomers'

                # Copie locale pour ne pas polluer le df global
                df_gen = df.copy()
                df_gen['Generation'] = df_gen[age_col].apply(get_generation)
                
                # On exclut les inconnus pour le tableau d'analyse pur
                df_clean_gen = df_gen[df_gen['Generation'] != 'Unknown / Error']

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
                            'Generation': gen,
                            'Customer Volume': df_g['anonymized_card_code'].nunique(),
                            'Brands / Basket': tickets_g['nb_brands'].mean(),
                            'Multi-Brand Rate (%)': (tickets_g['nb_brands'] > 1).mean() * 100,
                            'Average Basket (€)': tickets_g['spend'].mean()
                        })

                # 4. Formatage du Tableau de Synthèse
                df_results = pd.DataFrame(gen_stats)
                
                # Ajout de la ligne Baseline
                baseline_row = pd.DataFrame([{
                    'Generation': '🌍 GLOBAL BASELINE',
                    'Customer Volume': df['anonymized_card_code'].nunique(),
                    'Brands / Basket': global_avg_brands,
                    'Multi-Brand Rate (%)': global_multi_brand_rate,
                    'Average Basket (€)': global_avg_spend
                }])
                
                df_results = pd.concat([baseline_row, df_results], ignore_index=True)
                
                # Arrondir pour faire propre
                df_results['Brands / Basket'] = df_results['Brands / Basket'].round(2)
                df_results['Multi-Brand Rate (%)'] = df_results['Multi-Brand Rate (%)'].round(1)
                df_results['Average Basket (€)'] = df_results['Average Basket (€)'].round(2)

                # Affichage
                st.markdown("**1. Stratified Comparison (Benchmark vs Baseline)**")
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # 5. Graphique Visuel de Dépense
                st.markdown("**2. Variance from Global Average Basket**")
                df_plot = df_results[df_results['Generation'] != '🌍 GLOBAL BASELINE'].copy()
                df_plot['Baseline Variance (€)'] = df_plot['Average Basket (€)'] - global_avg_spend
                df_plot['Color'] = df_plot['Baseline Variance (€)'].apply(lambda x: '#118D57' if x > 0 else '#E50000')
                
                fig_gen = px.bar(df_plot, x='Generation', y='Baseline Variance (€)', 
                                 text_auto='.2f', color='Color', color_discrete_map='identity',
                                 title="Average Basket Over/Under-performance vs Global")
                fig_gen.update_layout(plot_bgcolor='white', margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig_gen, use_container_width=True)

                # 6. Le Warning de "Data Mastery" pour le Jury
                st.warning("""
                **⚠️ Methodological Alert (Data Mastery):** Although these insights are interesting for content marketing, **we advise against using age as the primary variable for the recommendation engine**. Our audits reveal that demographic data is missing or unreliable for nearly 30% of the base. This is why our main algorithm (Affinity Engine Tab) relies on RFM and actual purchasing behavior, which are 100% reliable.
                """)
        else:
            st.error("The 'age' column cannot be found in the dataset. Please check the demographic column name.")
# ==========================================
    # TAB 8 : LTV PREDICTION (LIFETIME VALUE)
    # ==========================================
with tab8:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Customer Value Predictive Model (3-Year LTV)**")
        st.write("This algorithm estimates the Revenue a customer will generate over the next 3 years based on their current behavior (Average Basket x Frequency x Margin).")
        st.markdown("</div>", unsafe_allow_html=True)

        col_param, col_calc = st.columns([1, 1.5], gap="large")
        
        with col_param:
            st.markdown("**1. Select Customer Profile**")
            liste_segments = sorted(df['RFM_Name'].dropna().unique().tolist())
            ltv_segment = st.selectbox("Select the segment to predict its future value:", liste_segments, key="ltv_box")
            
            st.markdown("**2. Financial Hypothesis**")
            marge_brute = st.slider("Estimated Sephora Gross Margin (%):", min_value=30, max_value=80, value=50, step=5)
            
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
                
                st.markdown(f"### Projection for segment: {ltv_segment}")
                
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='sephora-card' style='border-left: 4px solid #000;'><div class='metric-title'>Gross Future Value (3-Year Rev)</div><div class='metric-value'>{ltv_revenue:.2f} € / customer</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='sephora-card' style='border-left: 4px solid #E50000;'><div class='metric-title'>Estimated Net Profit (Actual LTV)</div><div class='metric-value'>{ltv_profit:.2f} € / customer</div></div>", unsafe_allow_html=True)
                
                st.info(f"💡 **Insight:** Sephora can afford to spend up to **{ltv_profit * 0.2:.2f} €** in Customer Acquisition Cost (CAC) or CRM gifts to recruit/retain a customer from this segment, while remaining highly profitable.")
    
    # ==========================================
    # TAB 9 : R.O.I SIMULATOR (BUSINESS CASE)
    # ==========================================
with tab9:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Financial Impact Simulator (Business Case)**")
        st.write("Estimate the financial gains (Uplift) generated by implementing our AI recommendations.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Calcul des bases de référence
        df_vip = df[df['RFM_Name'] == '1 - VIP']
        df_new = df[df['RFM_Name'] == '4 - New 3M']
        
        ca_actuel = df['salesVatEUR'].sum()
        panier_vip = df_vip['salesVatEUR'].sum() / df_vip['anonymized_card_code'].nunique() if not df_vip.empty else 200
        panier_new = df_new['salesVatEUR'].sum() / df_new['anonymized_card_code'].nunique() if not df_new.empty else 50
        
        col_sliders, col_kpi = st.columns([1, 1.5], gap="large")
        
        with col_sliders:
            st.markdown("**1. Performance Hypotheses (CRM Levers)**")
            st.info("Adjust the sliders according to the targets set for the teams:")
            
            conversion_rate = st.slider(
                "Conversion Rate (New ➡️ VIP):", 
                min_value=0.0, max_value=15.0, value=2.0, step=0.5,
                help="Percentage of New customers converted thanks to the right Gateway Brand."
            )
            
            anti_churn_rate = st.slider(
                "Anti-Churn Rescue (VIP):", 
                min_value=0.0, max_value=10.0, value=1.5, step=0.5,
                help="Inactive VIPs reactivated thanks to the Affinity Engine."
            )

        with col_kpi:
            st.markdown("**2. Annual Growth Projection**")
            
            nb_new = df_new['anonymized_card_code'].nunique() if not df_new.empty else 0
            nb_vip_churners = int(df_vip['anonymized_card_code'].nunique() * 0.15)
            
            gain_conversion = (nb_new * (conversion_rate / 100)) * (panier_vip - panier_new)
            gain_retention = (nb_vip_churners * (anti_churn_rate / 100)) * panier_vip
            
            uplift_total = gain_conversion + gain_retention
            croissance = (uplift_total / ca_actuel) * 100 if ca_actuel > 0 else 0

            st.markdown(f"""
            <div class="sephora-card" style="border: 2px solid #118D57; background-color: #F8FFF9 !important; text-align: center; padding: 30px;">
                <h3 style="color: #118D57 !important; margin:0;">💰 Potential Gain (Uplift)</h3>
                <h1 style="color: #118D57 !important; font-size: 48px; margin: 10px 0;">+ {uplift_total:,.0f} €</h1>
                <p style="color: #666; font-size: 16px; margin:0;">Representing an additional growth of <b>+{croissance:.2f}%</b> on the analyzed sample.</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='sephora-card' style='border-left: 4px solid #000;'><div class='metric-title'>Gain via Acquisition</div><div class='metric-value'>+ {gain_conversion:,.0f} €</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='sephora-card' style='border-left: 4px solid #E50000;'><div class='metric-title'>Gain via Retention</div><div class='metric-value'>+ {gain_retention:,.0f} €</div></div>", unsafe_allow_html=True)

    # ==========================================
    # TAB 10 : DATA AUDIT & DICTIONARY
    # ==========================================
with tab10:
        st.markdown("<div class='sephora-card'>", unsafe_allow_html=True)
        st.markdown("**Imported Dataset Health Diagnostic**")
        st.write("This tab analyzes the data quality of the provided sample in real-time.")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 1. APERÇU GLOBAL (Métriques de base) ---
        st.markdown("**1. Global Volumetrics**")
        nb_lignes = df.shape[0]
        nb_colonnes = df.shape[1]
        nb_clients_uniques = df['anonymized_card_code'].nunique() if 'anonymized_card_code' in df.columns else 0
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='sephora-card' style='border-left: 4px solid #000;'><div class='metric-title'>Total Rows (Transactions)</div><div class='metric-value'>{nb_lignes:,}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='sephora-card' style='border-left: 4px solid #000;'><div class='metric-title'>Total Columns (Variables)</div><div class='metric-value'>{nb_colonnes}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='sephora-card' style='border-left: 4px solid #E50000;'><div class='metric-title'>Unique Customers</div><div class='metric-value'>{nb_clients_uniques:,}</div></div>", unsafe_allow_html=True)

        # --- 2. AUDIT DES VALEURS MANQUANTES (Calculé en live) ---
        col_gauche, col_droite = st.columns([1, 1], gap="large")
        
        with col_gauche:
            st.markdown("**2. Missing Values Rate (Missingness)**")
            # Calcul des valeurs manquantes
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Empty Values']
            missing_data['% Missing'] = (missing_data['Empty Values'] / nb_lignes) * 100
            
            # On ne garde que les colonnes qui ont des trous, triées par ordre décroissant
            missing_data = missing_data[missing_data['Empty Values'] > 0].sort_values(by='% Missing', ascending=False)
            
            if not missing_data.empty:
                # Formatage pour un bel affichage
                missing_data['% Missing'] = missing_data['% Missing'].map('{:.1f}%'.format)
                st.dataframe(missing_data, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No missing values detected in this dataset!")

        # --- 3. DICTIONNAIRE DES DONNÉES (Pour le Jury) ---
        with col_droite:
            st.markdown("**3. Key Variables Dictionary**")
            
            dico_data = {
                "Column Name": [
                    "anonymized_card_code", 
                    "RFM_Segment_ID", 
                    "salesVatEUR", 
                    "anonymized_Ticket_ID", 
                    "brand", 
                    "Axe_Desc",
                    "age / gender"
                ],
                "Definition (Sephora Business)": [
                    "Unique customer identifier (Hashed for GDPR).",
                    "Loyalty score (1 = VIP, 2 = Good, 3 = Opportunist, 4 = New).",
                    "Revenue generated by the purchase line (in Euros).",
                    "Basket/receipt identifier (Groups products bought together).",
                    "Brand of the purchased product.",
                    "Global category (Make Up, Skincare, Fragrance, Haircare).",
                    "Demographic data (Warning: often missing or erroneous)."
                ]
            }
            df_dico = pd.DataFrame(dico_data)
            st.dataframe(df_dico, use_container_width=True, hide_index=True)

        # --- 4. ALERTE QUALITÉ (Bonus "Data Rigor") ---
        st.markdown("**4. Data Quality Warnings**")
        st.warning("""
        **Observations from our Exploratory Analysis (Deliverable 1):**
        - **Cold Start Problem:** Over 74% of the data related to the *first purchase* is missing. We pivoted our analysis to recent history rather than original acquisition.
        - **Demographic Bias:** Nearly 28% of data missing on age and gender. This is why our recommendation algorithm (Brand Affinity) relies exclusively on purchasing behavior (RFM) rather than socio-demographic profiles.
        """)