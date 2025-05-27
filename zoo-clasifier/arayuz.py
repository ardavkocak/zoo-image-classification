import timm
import torch
import torchvision.transforms as T
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os
import torch.nn.functional as F

# --- Model Yükleme ve Hazırlık ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Görsel Dönüşümü
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# Sınıf İsimleri
#val_dir = "zoo-clasifier/animal/val"  # bu klasörde her sınıfın bir klasörü varsa
#class_names = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']


# Model Yükle
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load('zoo-clasifier/vit_model.pth', map_location=device))
model.to(device)
model.eval()


# --- Tahmin Fonksiyonu ---
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_names[predicted_index]
        confidence = probabilities[0][predicted_index].item()

    return predicted_class, confidence, probabilities[0].cpu().numpy()


def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    # Görseli göster
    img = Image.open(file_path).resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    # Tahmin yap
    predicted_class, confidence, all_probs = predict_image(file_path)

    result_label.config(
        text=f"Tahmin: {predicted_class}\nGüven Skoru: {confidence:.4f}"
    )

    #  En yüksek 6 sınıfı al
    import numpy as np
    topk = 6
    top_indices = np.argsort(all_probs)[-topk:][::-1]  # en büyükten küçük olasılığa
    top_labels = [class_names[i] for i in top_indices]
    top_probs = [all_probs[i] for i in top_indices]
    top_colors = [plt.cm.Paired.colors[i % len(plt.cm.Paired.colors)] for i in top_indices]
    explode = [0.1 if class_names[i] == predicted_class else 0 for i in top_indices]

    # Pie chart
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        top_probs,
        labels=top_labels,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        colors=top_colors,
        textprops={'fontsize': 11}
    )

    # Başlık
    ax.set_title("En Yüksek 6 Tahminin Dağılımı", fontsize=14, fontweight='bold')

    # Legend (sağda açıklama)
    ax.legend(
        wedges,
        top_labels,
        title="Sınıflar",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        title_fontsize=11
    )

    plt.axis('equal')  # Yuvarlak şekli koru
    plt.tight_layout()
    plt.show()




# --- Tkinter Arayüz ---
root = tk.Tk()
root.title("Hayvan Görsel Sınıflandırma")
root.geometry("400x400")

btn = tk.Button(root, text="📂 Görsel Seç ve Test Et", command=open_image)
btn.pack(pady=10)

panel = tk.Label(root)  # Görseli gösterecek alan
panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
