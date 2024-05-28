import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

all_data = pd.read_csv("alldata.csv")

@st.cache_data
def prepare_data(all_data):
    # Copy dataset for manipulation
    combined_df = all_data.copy()
    
    # Create a new column that combines "nama" and "kategori"
    combined_df['nama_kategori'] = combined_df['nama'] + ' - ' + combined_df['kategori']
    
    # Use TF-IDF Vectorizer with English stop words removal
    vectorizer = TfidfVectorizer(stop_words='english')
    X1 = vectorizer.fit_transform(combined_df["nama_kategori"])
    
    # Fit K-Means to the dataset
    kmeans = KMeans(n_clusters=15, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X1)
    
    # Initialize K-Means model
    true_k = 15
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=42)
    
    # Fit model to data
    model.fit(X1)
    
    # Get cluster labels for each sample
    cluster_labels = model.predict(X1)
    
    # Add cluster labels to DataFrame
    combined_df['cluster'] = cluster_labels
    
    return combined_df, vectorizer, model

# Prepare the data for recommendations
random_sample, vectorizer, model = prepare_data(all_data)

@st.cache_data
# Function to show recommendations
def show_recommendations(product):
    # Transformasi produk input ke dalam bentuk vektor
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    cluster_id = prediction[0]

    # Filter data berdasarkan cluster_id
    cluster_data = random_sample[random_sample['cluster'] == cluster_id]

    # Mengambil TF-IDF matrix dari produk dalam klaster yang sama
    cluster_tfidf_matrix = vectorizer.transform(cluster_data['nama_kategori'])

    # Menghitung kesamaan kosinus antara produk input dan produk dalam klaster yang sama
    cosine_similarities = cosine_similarity(Y, cluster_tfidf_matrix).flatten()

    # Mengurutkan produk berdasarkan kesamaan kosinus dan mengambil 10 produk teratas
    similar_indices = cosine_similarities.argsort()[:-11:-1]
    similar_items = cluster_data.iloc[similar_indices]

    # Mengurutkan produk berdasarkan harga_per_unit terendah
    similar_items = similar_items.sort_values(by='harga_pound')
    return similar_items

@st.cache_data
# Function to recommend top selling products
def recommend_top_selling_products(combined_df, top_n=10):
    top_selling_products = combined_df.groupby('nama').agg({'harga_pound': 'sum'}).reset_index()
    top_selling_products = top_selling_products.sort_values(by='harga_pound', ascending=False)
    top_selling_products = top_selling_products.merge(combined_df[['nama', 'nama_toko', 'harga_per_unit', 'unit', 'kategori']], on='nama').drop_duplicates()
    return top_selling_products.head(top_n)

@st.cache_data
# Function to recommend best priced products
def recommend_best_priced_products(combined_df, top_n=10):
    average_price_per_unit = combined_df['harga_per_unit'].mean()
    best_priced_products = combined_df[combined_df['harga_per_unit'] < average_price_per_unit]
    best_priced_products = best_priced_products.drop_duplicates(subset=['nama']).sort_values(by='harga_per_unit')
    return best_priced_products.head(top_n)

@st.cache_data
# Function to recommend random products
def recommend_random_products(combined_df, top_n=10):
    random_products = combined_df.sample(n=top_n)
    return random_products

@st.cache_data
# Fungsi rekomendasi produk berdasarkan kategori terpopuler
def recommend_cheapest_products_by_category(combined_df):
    unique_categories = combined_df['kategori'].unique()
    print("\nKategori yang tersedia:")
    for idx, category in enumerate(unique_categories, start=1):
        print(f"{idx}. {category}")

    try:
        category_choice = int(input("\nMasukkan nomor kategori yang Anda pilih: "))
        if 1 <= category_choice <= len(unique_categories):
            selected_category = unique_categories[category_choice - 1]
            category_products = combined_df[combined_df['kategori'] == selected_category]
            cheapest_products = category_products.sort_values(by='harga_per_unit').head(10)
            if not cheapest_products.empty:
                print(f"\nRekomendasi Produk dengan Harga Per Unit Termurah dari Kategori: {selected_category}")
                display_table(cheapest_products[['nama_toko', 'nama', 'harga_per_unit', 'unit', 'kategori', 'harga_per_unit']])
            else:
                print(f"Tidak ada produk yang ditemukan dalam kategori '{selected_category}'.")
        else:
            print("Pilihan kategori tidak valid.")
    except ValueError:
        print("Masukkan harus berupa angka.")

def display_table(df):
    st.table(df)

# Streamlit app
st.title("Product Recommendation System")

menu = ["Home", "Recommendation by Name", "Top Selling Products", "Best Priced Products", "Random Products", "Cheapest Products by Category"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Home")
    st.subheader("Information")
    st.markdown('''
    UMKM have an important role in the national economy, namely as the main contributors to Gross Domestic Product and job creation. Based on data in 2024 there will be 83.3 million registered and unregistered UMKM, some of which have similar or even the same products. 
    
    :cherry_blossom: Pricing is important for UMKM to be able to compete better with imported products and to compete with products between UMKM       
    
    :cherry_blossom: A system that can predict product trends that will be popular helps UMKM manage stock more efficiently           
    
    Having the 2 things above can help UMKM increase sales and attract consumers
                ''')
    st.subheader("The Data Set")
    st.markdown("This dataset consists of product sales information in various stores. There are five DataFrames, each of which contains data related to price, unit, product name, category and other relevant information. The following is an explanation of each column in this dataset: ")

    st.markdown("**nama_toko:** Name of the shop where the product is sold. This column contains information about the different store names.")
    st.markdown("**harga_pound:** Product price in pounds. This column shows the total price of the product.")
    st.markdown("**harga_per_unit:** Price per unit of product. It shows the price per unit of the product sold.")
    st.markdown("**unit:** The units of the product sold (for example, liters, kilograms, pcs). This column contains information about the unit of measurement of the product.")
    st.markdown("**nama:** Product name. This column contains the names of the products sold.")
    st.markdown("**kategori:** Category of the product. It covers various categories like food, drinks, household necessities, etc.")
    st.markdown("**brand_sendiri:** Indicator of whether the product is a shop's own brand (private label). The value is a boolean (True or False).")
    st.markdown("**tanggal:** The date the sales or price data was recorded. This column shows when the data was taken.")
                
    st.dataframe(all_data.sample(15))

elif choice == "Recommendation by Name":
    st.subheader("Recommendation by Product Name")
    product = st.text_input("Enter product name")
    if st.button("Show Recommendations"):
        recommendations = show_recommendations(product)
        if not recommendations.empty:
            st.write("Recommended Products:")
            for index, row in recommendations.iterrows():
                st.write(f"Nama Toko: {row['nama_toko']}")
                st.write(f"Nama Barang: {row['nama']}")
                st.write(f"Harga Pound: {row['harga_pound']}")
                st.write(f"Kategori: {row['kategori']}")
                st.write("----------------------------------")
        else:
            st.write("No recommendations found.")

elif choice == "Top Selling Products":
    st.subheader("Top Selling Products")
    top_selling_products = recommend_top_selling_products(all_data)
    st.table(top_selling_products)

elif choice == "Best Priced Products":
    st.subheader("Best Priced Products")
    best_priced_products = recommend_best_priced_products(all_data)
    st.table(best_priced_products)

elif choice == "Random Products":
    st.subheader("Random Products")
    random_products = recommend_random_products(all_data)
    st.table(random_products)

elif choice == "Cheapest Products by Category":
    st.subheader("Cheapest Products by Category")
    unique_categories = all_data['kategori'].unique()
    category = st.selectbox("Choose a category", unique_categories)
    if category:
        category_products = all_data[all_data['kategori'] == category]
        cheapest_products = category_products.sort_values(by='harga_per_unit').head(10)
        display_table(cheapest_products)
