import requests
import json
import os
import time
from sentence_transformers import SentenceTransformer, util

BASE_URL = "https://model.cosmosytu.com/cosmosvlm/vlmrecords"
SAVE_DIR = "images"
JSON_FILENAME = "output.json"
LOGIN_URL = "https://model.cosmosytu.com/cosmosvlm/login?next=/cosmosvlm/vlmrecords"  
MAX_IMAGES = 100 ##denemek için 

with open("password.txt", "r") as file:
    PASSWORD = file.read().strip()
start_time = time.perf_counter()

session = requests.Session()

login_data = {
    "password": PASSWORD
}
response = session.post(LOGIN_URL, data=login_data)

if response.status_code == 200:
    print("Giriş başarılı!")
else:
    print("Giriş başarısız!")
    exit()


data_url = "https://model.cosmosytu.com/cosmosvlm/vlmrecords"  
response = session.get(data_url)

if response.status_code != 200:
    print("Veriler alınamadı!")
    exit()

data = response.json()
previous_descriptions = [] 
output_data = []

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
i = 0
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #farklı modeller ve yaklaşımlar denenebilir
image_count = 0

print("toplam veri sayısı")
print(len(data["records"]))
for record in data["records"]:
    i+=1
    if image_count >= MAX_IMAGES:
        print(f"Maksimum {MAX_IMAGES} görsel kaydedildi. İşlem tamamlandı!")
        break
    
    description = record["description"]
    
    if description.startswith("Üzgünüm"):
        continue
    
    if previous_descriptions:
       
        current_embedding = model.encode(description, convert_to_tensor=True)

        last_two_descriptions = previous_descriptions[-2:] if len(previous_descriptions) >= 2 else previous_descriptions
        last_two_embeddings = model.encode(last_two_descriptions, convert_to_tensor=True)
        

        skip = False
        for last_embedding in last_two_embeddings:
            similarity = util.cos_sim(current_embedding, last_embedding).item()
            if similarity >= 0.8: #farklı bir benzerlik metriği de kullanılabilir
                print("Benzerlik yüksek, atlandı.")
                skip = True
                break
        
        if skip:
            continue

    
    image_url = f"https://model.cosmosytu.com/cosmosvlm/show/{record['id']}"
    image_path = os.path.join(SAVE_DIR, record["filename"])
    
    image_response = session.get(image_url)
    if image_response.status_code == 200:
        with open(image_path, "wb") as img_file:
            img_file.write(image_response.content)
        print(f"{record['filename']} kaydedildi.")
        image_count += 1
    else:
        print(f"{record['filename']} indirilemedi.")
    

    output_data.append({
        "prompt": "Bu resimde gördüklerini anlat",
        "chosen": "Deneme Yanıtı",
        "rejected": description,
        "image": record["filename"]
    })
    
    # Description'ı önceki description listesine ekle
    previous_descriptions.append(description)
print(f"İncelenen veri sayısı:{i}")

# JSON dosyasını kaydet
with open(JSON_FILENAME, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

print(f"Tüm işlemler tamamlandı. {JSON_FILENAME} dosyası oluşturuldu.")
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Toplam çalışma süresi: {elapsed_time:.2f} saniye")
print(f"Görsel başına Düşen süre: {elapsed_time/i:.2f} saniye")
