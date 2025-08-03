from detect import detect_image

img_path = "uploads/hilalsafar1446(4).jpg"  # hanya 1 file
output_img, output_csv = detect_image(img_path)

print("Hasil gambar disimpan di:", output_img)
print("Hasil CSV disimpan di:", output_csv)
