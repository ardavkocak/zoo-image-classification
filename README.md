# Zoo Image Classification

Vision Transformer (ViT) tabanlı, 90 farklı hayvan türünü ayırt eden bir görsel sınıflandırma uygulaması. Kullanıcı arayüzü Tkinter ile yazılmış olup, seçilen bir görsel üzerinde tahmin yapar ve en yüksek olasılığa sahip 6 sınıfı pasta grafiği ile görselleştirir.

## Özellikler

- **90 sınıflık hayvan tanıma**: antilop, ayı, kelebek, kedi, köpek, fil, aslan, kaplan, balina, zebra ve daha fazlası
- **Vision Transformer (ViT-Base, 16x16 patch)** mimarisi (`timm` üzerinden)
- **Tkinter arayüzü**: tek tıkla görsel yükle, anlık tahmin al
- **Top-6 olasılık görselleştirmesi**: matplotlib pasta grafiği ile sınıf dağılımı
- GPU varsa CUDA, yoksa CPU üzerinde otomatik çalışır

## Gereksinimler

```bash
pip install torch torchvision timm pillow matplotlib numpy
```

Ayrıca eğitilmiş model ağırlıkları gerekir: `zoo-clasifier/vit_model.pth` (repoda yer almaz, ayrı sağlanmalıdır).

## Çalıştırma

```bash
cd zoo-clasifier
python arayuz.py
```

Açılan pencerede **"📂 Görsel Seç ve Test Et"** butonuna tıklayın, bir `.jpg/.png/.jpeg` dosyası seçin. Uygulama:

1. Seçilen görseli pencerede gösterir
2. Tahmin edilen sınıfı ve güven skorunu yazar
3. En yüksek 6 sınıfın olasılık dağılımını pasta grafiği ile açar

## Proje Yapısı

```
zoo-clasifier/
├── arayuz.py          # Tkinter arayüzü + tahmin mantığı
├── vit_model.pth      # Eğitilmiş ViT ağırlıkları (harici)
└── Figure_*.png       # Örnek çıktı görselleri
```

## Model Detayları

- **Mimari**: `vit_base_patch16_224` (timm)
- **Girdi**: 224x224 RGB, normalize edilmiş ([0.5, 0.5, 0.5] mean/std)
- **Çıktı**: 90 sınıf için softmax olasılıkları
