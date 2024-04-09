from PIL import Image
import os
# Verzeichnis mit Bildern
image_folder = 'Figure_1'

# Dateinamen der Bilder
images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".png")]

# Bilder in der richtigen Reihenfolge sortieren
images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# GIF erstellen
gif_path = 'Figure_1/Figure_1.gif'

# Bilder öffnen und zu GIF hinzufügen
frames = []
for image in images:
    frames.append(Image.open(image))

# GIF speichern
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)

print(f"GIF erstellt: {gif_path}")
