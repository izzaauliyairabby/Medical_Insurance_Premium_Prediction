# MLT Proyek Pertama | Medical Insurance Premium Prediction

###### Disusun oleh : Izza Auliyai Rabby

Ini adalah proyek pertama analisis prediktif untuk memenuhi submission Dicoding Kelas Machine Learning Terapan. 

Proyek ini membangun model *machine learning* yang dapat memprediksi biaya pertanggungan medis tahunan.

## 1. Project Domain

### Latar Belakang

Proyek "Medical Insurance Premium Prediction" berfokus pada industri asuransi kesehatan, yang menyediakan perlindungan finansial terhadap risiko kesehatan bagi individu dan keluarga. Dalam industri ini, perusahaan asuransi menawarkan layanan kesehatan dengan mengenakan premi kepada pemegang polis.

Relevansi proyek ini dalam konteks asuransi kesehatan adalah:

Penentuan Premi yang Lebih Akurat: Model prediktif yang dikembangkan bertujuan membantu perusahaan asuransi dalam menetapkan premi berdasarkan risiko spesifik pemegang polis, seperti usia, riwayat penyakit, riwayat keluarga, serta kondisi fisik seperti tinggi dan berat badan. Dengan pendekatan ini, premi lebih mencerminkan tingkat risiko individual.

Keadilan dalam Penetapan Premi: Model ini memastikan premi yang dikenakan lebih adil karena memperhitungkan faktor risiko secara komprehensif. Premi yang dihasilkan sesuai dengan tingkat risiko kesehatan, sehingga setiap pemegang polis membayar sesuai dengan risiko yang mereka hadapi.

Manfaat Bagi Perusahaan Asuransi:

Penentuan Premi Lebih Tepat: Model ini membantu perusahaan menetapkan premi secara lebih akurat, sehingga risiko finansial lebih terkendali.
Peningkatan Efisiensi: Dengan menggunakan teknologi analitik, perusahaan dapat mempercepat proses penentuan premi dan mengurangi penggunaan sumber daya.
Manfaat Bagi Calon Pemegang Polis:

Premi yang Lebih Adil: Premi dihitung berdasarkan risiko individu, yang memberikan pemahaman lebih jelas dan rasa keadilan bagi calon pemegang polis.
Perlindungan Finansial yang Tepat: Dengan premi yang sesuai dengan kebutuhan dan risiko kesehatan, pemegang polis mendapatkan perlindungan finansial yang lebih optimal.
Secara keseluruhan, proyek ini menawarkan solusi win-win: perusahaan asuransi dapat mengoptimalkan penentuan premi dan mengurangi risiko keuangan, sementara pemegang polis menerima premi yang lebih sesuai dengan profil risiko mereka.

## 2. Business Understanding

Proyek ini dirancang untuk perusahaan asuransi kesehatan dengan karakteristik bisnis sebagai berikut:

Perusahaan Asuransi Kesehatan Berbasis Analitik: Perusahaan ini memiliki fokus kuat pada penggunaan data untuk pengambilan keputusan yang lebih baik. Mereka memahami pentingnya analitik prediktif dan machine learning dalam menentukan premi asuransi kesehatan yang lebih tepat dan didasarkan pada risiko individual pelanggan.

Perusahaan dengan Data Pelanggan yang Luas: Perusahaan ini memiliki akses ke data pelanggan yang lengkap dan rinci. Informasi yang dimiliki mencakup variabel seperti usia, riwayat kesehatan, operasi sebelumnya, riwayat keluarga, serta data fisik seperti tinggi dan berat badan. Data yang kaya ini akan digunakan untuk membangun model prediktif yang kuat.

Perusahaan yang Berfokus pada Keunggulan Kompetitif: Perusahaan ini berorientasi untuk tetap unggul di pasar asuransi kesehatan dengan memanfaatkan teknik analitik prediktif dan machine learning guna mengoptimalkan proses penetapan premi. Dengan demikian, mereka dapat menawarkan produk asuransi yang lebih kompetitif kepada calon pelanggan.

Dengan memperhatikan karakteristik perusahaan ini, proyek "Medical Insurance Premium Prediction" disesuaikan untuk membantu perusahaan mencapai tujuan bisnis mereka, dengan cara-cara berikut:

Premi yang Lebih Tepat: Model prediktif yang dikembangkan dalam proyek ini akan memungkinkan perusahaan menentukan premi yang lebih akurat berdasarkan berbagai faktor risiko, seperti usia, riwayat medis, operasi sebelumnya, riwayat keluarga, dan kondisi fisik lainnya. Estimasi premi yang lebih tepat ini membantu perusahaan membuat keputusan berbasis data yang lebih informatif, sekaligus meminimalkan potensi risiko finansial.

Meningkatkan Efisiensi dan Produktivitas: Dengan penerapan analitik prediktif dan machine learning, perusahaan dapat meningkatkan efisiensi dalam menetapkan premi. Proses ini akan mengotomatisasi pemrosesan data dan menghasilkan perkiraan premi secara lebih cepat, mengurangi ketergantungan pada metode manual yang memakan waktu. Ini akan menghemat sumber daya perusahaan dan memungkinkan mereka lebih fokus pada aspek penting bisnis lainnya.

Pengambilan Keputusan yang Lebih Baik: Model prediktif akan memberikan wawasan lebih mendalam tentang faktor-faktor risiko yang memengaruhi premi asuransi kesehatan. Dengan wawasan ini, perusahaan dapat lebih memahami pelanggan mereka, mengidentifikasi tren, dan membuat keputusan strategis yang lebih baik. Ini juga membantu mereka menyusun strategi yang lebih efektif untuk memperkuat posisi mereka di pasar.

Keunggulan Kompetitif di Pasar: Dengan model analitik prediktif yang lebih akurat, perusahaan dapat menarik lebih banyak calon pemegang polis dengan menawarkan premi yang lebih adil dan produk yang lebih kompetitif. Hal ini akan membantu perusahaan membedakan diri dari pesaing, memperbesar pangsa pasar, dan membangun hubungan jangka panjang yang kuat dengan pelanggan.

Melalui implementasi model prediktif ini, perusahaan asuransi kesehatan dapat membuat keputusan yang lebih cerdas, meningkatkan efisiensi, memperoleh keunggulan di pasar, serta memberikan layanan yang lebih baik kepada calon pemegang polis.

## 3. Problem Statements

Langkah-langkah yang dapat diambil untuk meningkatkan akurasi dalam menentukan premi asuransi kesehatan meliputi:

Penggunaan Data yang Lebih Luas dan Kaya: Mengumpulkan dan menggunakan data yang lebih lengkap seperti riwayat medis, kondisi fisik, riwayat keluarga, serta faktor lingkungan akan membantu dalam menghasilkan perkiraan premi yang lebih akurat.

Penerapan Teknik Machine Learning yang Canggih: Memanfaatkan algoritma machine learning yang lebih canggih, seperti regresi, pohon keputusan, atau jaringan saraf, memungkinkan model mempelajari pola kompleks dari data dan memberikan perkiraan yang lebih tepat.

Analisis Faktor Risiko yang Mendetail: Dengan menganalisis secara komprehensif faktor risiko seperti usia, riwayat penyakit, dan kondisi medis, model dapat lebih akurat memperhitungkan risiko kesehatan individu dan menetapkan premi yang sesuai.

Pengujian dan Validasi Model: Menguji dan memvalidasi model secara berkala dengan data aktual akan membantu dalam menyesuaikan prediksi agar tetap relevan dan akurat seiring dengan perubahan kondisi kesehatan dan pasar.

Risiko Keuangan Bagi Perusahaan Asuransi:
Masalah: Jika premi ditetapkan secara tidak akurat, ada potensi ketidakseimbangan antara risiko yang dihadapi perusahaan dan premi yang diterima. Premi yang terlalu rendah untuk kelompok berisiko tinggi dapat menyebabkan klaim yang melebihi pendapatan, memicu kerugian finansial.

Contoh: Jika perusahaan menetapkan premi yang rendah untuk individu dengan risiko kesehatan tinggi, misalnya yang memiliki riwayat penyakit kronis, mereka mungkin menghadapi lebih banyak klaim dari yang diprediksi, yang dapat menyebabkan kerugian besar.

Dalam proyek "Medical Insurance Premium Prediction", faktor-faktor risiko ini akan dianalisis secara menyeluruh dan dimasukkan ke dalam model prediktif. Dengan memanfaatkan machine learning, model akan menghasilkan perkiraan premi yang lebih akurat dengan mempertimbangkan variabel-variabel penting yang relevan.

Selain itu, dengan mengidentifikasi faktor-faktor yang paling berpengaruh dalam penentuan premi, perusahaan asuransi dapat memberikan penjelasan yang lebih transparan kepada calon pemegang polis tentang alasan di balik besaran premi mereka. Ini akan meningkatkan transparansi dan kepercayaan pemegang polis.

## 4. Goals

Proyek ini memiliki tiga tujuan utama yang berfokus pada peningkatan akurasi prediksi premi, transparansi dalam penetapan premi, serta pengurangan risiko keuangan bagi perusahaan asuransi kesehatan. Metrik evaluasi yang sesuai digunakan untuk mengukur pencapaian setiap tujuan.

1. Mengembangkan Model Analisis Prediktif
Tujuan: Menghasilkan model prediktif yang dapat memperkirakan premi asuransi kesehatan dengan lebih akurat menggunakan teknik machine learning.
Metrik Evaluasi:
Akurasi Prediksi: Mengukur kemampuan model dalam memperkirakan premi dengan tepat. Akurasi yang lebih tinggi menandakan model tersebut efektif.
Mean Squared Error (MSE): Nilai MSE yang lebih rendah menunjukkan bahwa prediksi premi semakin dekat dengan nilai premi yang sebenarnya, mengindikasikan keberhasilan model dalam memperkirakan premi dengan tepat.
2. Meningkatkan Transparansi
Tujuan: Memberikan pemahaman yang lebih baik kepada calon pemegang polis mengenai faktor-faktor yang mempengaruhi penetapan premi, untuk membangun kepercayaan dan kepuasan pelanggan.
Metrik Evaluasi:
Tingkat Transparansi: Dapat diukur melalui survei atau wawancara dengan calon pemegang polis, untuk mengetahui sejauh mana mereka memahami alasan di balik besarnya premi yang mereka terima.
Tingkat Kepuasan: Survei kepuasan calon pemegang polis dapat digunakan untuk mengukur apakah peningkatan transparansi berhasil meningkatkan kepercayaan dan kepuasan pelanggan terhadap perusahaan.
3. Mengurangi Risiko Keuangan
Tujuan: Membantu perusahaan asuransi mengurangi risiko keuangan yang terkait dengan penetapan premi yang tidak akurat, menjaga stabilitas keuangan perusahaan.
Metrik Evaluasi:
Penurunan Klaim Berlebih: Membandingkan jumlah klaim asuransi sebelum dan setelah penerapan model prediktif. Penurunan klaim yang tidak sesuai dengan premi menandakan pengurangan risiko keuangan.
Performa Keuangan: Analisis kinerja keuangan perusahaan, seperti stabilitas laba dan pengelolaan risiko, dapat menunjukkan apakah perusahaan berhasil mengurangi risiko keuangan akibat premi yang tidak akurat.
Tambahan:
Tingkat Optimasi Proses: Dapat diukur melalui peningkatan efisiensi dalam proses penetapan premi yang lebih cepat dan otomatis dengan menggunakan model prediktif berbasis data.
Keberlanjutan Penggunaan Model: Mengukur sejauh mana perusahaan mengadopsi model dalam operasi sehari-hari dan seberapa baik model dapat diintegrasikan dengan sistem yang ada.
Dengan menggunakan metrik evaluasi yang relevan seperti akurasi prediksi, tingkat transparansi, dan pengurangan risiko keuangan, keberhasilan proyek ini dapat dinilai secara menyeluruh dan memberikan dampak yang signifikan bagi perusahaan asuransi dan calon pemegang polis.

## 5. Solution statements

Solusi untuk proyek prediksi premi asuransi kesehatan melibatkan beberapa tahapan strategis dan penggunaan algoritma machine learning yang dirancang untuk meningkatkan akurasi dan efisiensi. Berikut adalah rincian solusi yang diberikan:

1. Eksplorasi Data (Exploratory Data Analysis - EDA)
Tujuan: Memahami data secara mendalam sebelum model dilatih.
Langkah-langkah:
Menganalisis distribusi variabel seperti usia, jenis kelamin, kondisi kesehatan, gaya hidup, dan biaya historis klaim asuransi.
Melihat pola hubungan antara variabel-variabel kunci, seperti korelasi antara usia atau status merokok dengan premi.
Identifikasi dan penanganan outliers, data yang hilang, serta variabel-variabel yang mungkin sangat mempengaruhi prediksi.
Output: Wawasan yang akan memandu pemilihan fitur yang relevan dan teknik pra-pemrosesan data yang tepat.
2. Algoritma yang Digunakan
Support Vector Regression (SVR):
Kegunaan: Digunakan untuk memprediksi premi berdasarkan variabel-variabel yang relevan. SVR cocok untuk menangani data non-linear dengan menerapkan kernel yang tepat, seperti radial basis function (RBF).
Kelebihan: Memiliki kemampuan yang kuat untuk menemukan hubungan yang kompleks antara variabel-variabel prediktor dan premi, serta menangani data dengan distribusi yang tidak normal.
Huber Regressor:
Kegunaan: Algoritma regresi yang tangguh terhadap outliers, menjaga kestabilan prediksi meskipun ada data yang ekstrem.
Kelebihan: Menggabungkan kekuatan regresi linier dan ketahanan terhadap data yang mengandung noise, menghasilkan model yang lebih stabil dalam lingkungan data yang tidak sempurna.
3. Penggunaan Library PyCaret
Mengapa PyCaret?: PyCaret adalah library yang memfasilitasi proses pembangunan model machine learning dengan sangat cepat dan efisien, bahkan bagi proyek-proyek besar.
Fitur PyCaret:
Memungkinkan automated machine learning (AutoML) untuk membandingkan berbagai algoritma sekaligus.
Menyediakan berbagai algoritma regression, termasuk SVR, Huber Regressor, serta banyak opsi lain yang dapat dieksplorasi.
Dilengkapi dengan proses hyperparameter tuning untuk meningkatkan kinerja model lebih lanjut.
Hasil yang Diharapkan: Dengan menggunakan PyCaret, proses pemilihan algoritma yang optimal dapat dilakukan lebih cepat dengan hasil yang lebih akurat.
4. Evaluasi dengan Metrik MSE (Mean Squared Error)
MSE sebagai metrik utama untuk mengukur akurasi model.
Alasan Penggunaan MSE:
Menghitung rata-rata kuadrat selisih antara nilai yang diprediksi dan nilai aktual, memberikan indikasi seberapa jauh prediksi dari target sebenarnya.
Metrik ini menekankan kesalahan yang besar, sehingga model yang optimal adalah yang mampu meminimalkan kesalahan besar dalam prediksi premi.
Tujuan: Menghasilkan model dengan MSE yang rendah, menandakan akurasi prediksi yang tinggi.
Kesimpulan
Melalui pendekatan yang komprehensif, dari eksplorasi data hingga evaluasi model menggunakan metrik yang tepat, solusi ini diharapkan dapat:

Menghasilkan model prediktif yang mampu memperkirakan premi asuransi kesehatan secara akurat.
Meningkatkan transparansi dalam penentuan premi dengan mengidentifikasi faktor-faktor risiko yang paling signifikan.
Mengurangi risiko keuangan bagi perusahaan asuransi dengan menetapkan premi yang lebih tepat dan efisien.
Pemanfaatan PyCaret dan algoritma yang tepat seperti SVR dan Huber Regressor memastikan proses ini dapat diotomatisasi dan dioptimalkan, membawa hasil yang signifikan bagi perusahaan asuransi dan calon pemegang polis.

## 6. Data Understanding

Dataset yang digunakan dalam proyek ini merupakan data parameter terkait kesehatan yang diberikan hampir oleh 1000 konsumen secara sukarela.

Dataset dapat diunduh di: [Medical Insurance Premium Prediction](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction).

### Sample data
| Age | Diabetes | BloodPressureProblems | AnyTransplants | AnyChronicDiseases | Height | Weight | KnownAllergies | HistoryOfCancerInFamily | NumberOfMajorSurgeries | PremiumPrice |
|-----|----------|-----------------------|----------------|--------------------|--------|--------|----------------|-------------------------|-----------------------|--------------|
| count | 986.000000 | 986.000000 | 986.000000 | 986.000000 | 986.000000 | 986.000000 | 986.000000 | 986.000000 | 986.000000 | 986.000000 |
| mean | 41.745436 | 0.419878 | 0.468560 | 0.055781 | 0.180527 | 168.182556 | 76.950304 | 0.215010 | 0.117647 | 0.667343 | 24336.713996 |
| std | 13.963371 | 0.493789 | 0.499264 | 0.229615 | 0.384821 | 10.098155 | 14.265096 | 0.411038 | 0.322353 | 0.749205 | 6248.184382 |
| min | 18.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 145.000000 | 51.000000 | 0.000000 | 0.000000 | 0.000000 | 15000.000000 |
| 25% | 30.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 161.000000 | 67.000000 | 0.000000 | 0.000000 | 0.000000 | 21000.000000 |
| 50% | 42.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 168.000000 | 75.000000 | 0.000000 | 0.000000 | 1.000000 | 23000.000000 |
| 75% | 53.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 176.000000 | 87.000000 | 0.000000 | 0.000000 | 1.000000 | 28000.000000 |
| max | 66.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 188.000000 | 132.000000 | 1.000000 | 1.000000 | 3.000000 | 40000.000000 |

Berikut informasi pada dataset :

- Dataset memiliki format CSV (Comma-Seperated Values).
- Dataset memiliki 986 sample dengan 11 fitur.
- Dataset memiliki 11 fitur bertipe int64.
- Tidak ada missing value dalam dataset.

### Variabel-variabel pada Medical Insurance Premium Prediction dataset adalah sebagai berikut:
- Age : merupakan rentang umur dari pasien tersebut
- Diabetes : apakah pasien tersebut memiliki kadar gula darah abnormal atau tidak
- BloodPressureProblems: apakah pasien tersebut memiliki tingkat tekanan darah abnormal
- AnyTransplants: pernah melakukan transplantasi organ utama apapun
- AnyChronicDiseases: apakah pasien menderita penyakit kronis seperti asthama, dll
- Height: rentang tinggi badan pasien
- Weight: rentang berat badan pasien
- KnownAllergies: apakah pasien memiliki alergi yang diketahui
- HistoryOfCancerInFamily: apakah ada relatif darah dari pasien yang mengidap bentuk kanker apa pun
- NumberOfMajorSurgeries: jumlah operasi besar yang dijalani pasien tersebut
- PremiumPrice: harga premium tahunan

### Pendalaman Data Understanding
- Melakukan tahapan EDA seperti mendeskripsikan variabel, mencari *outlier*s, Univariate hingga Multi-variate analysis.
- Untuk menganalisa *outlier*s bisa menggunaka boxplot dengan memanggil fungsi .plot() pada pandas
- Mengecek data missing value dan membersihkan data missing value dengan membuat simple logic program
- Menggunakan histogram untuk melihat penyebaran data dengan *library* pandas fungsi .hist()
- Mencari keterkaitan antar fitur numerik dan fitur kategori dengan correlation matrix menggunakan fungsi pandas dan visualisasi heatmap dengan seaborn

### Visualisasi proses Data Understanding

Untuk mengatasi outlier, salah satu metode yang umum digunakan adalah metode IQR (Interquartile Range) dengan visualisasi menggunakan boxplot. Berikut penjelasan mengenai metode IQR dan visualisasi boxplot:

Metode IQR:
1. Konsep: IQR merupakan ukuran statistik yang menggambarkan rentang atau sebaran data pada bagian tengah distribusi data. IQR dihitung dengan mengurangi nilai kuartil ketiga (Q3) dengan nilai kuartil pertama (Q1). Outlier dianggap sebagai nilai yang terletak di luar rentang IQR yang ditentukan.
2. Cara Kerja:
   - Hitung Q1 (kuartil pertama) dan Q3 (kuartil ketiga) dari data.
   - Hitung IQR dengan mengurangi Q1 dari Q3.
   - Tentukan batas atas dan batas bawah untuk outlier dengan menggunakan rumus: batas atas = Q3 + (1.5 * IQR), batas bawah = Q1 - (1.5 * IQR).
   - Data yang berada di luar batas atas dan batas bawah tersebut dianggap sebagai outlier.

Visualisasi Boxplot:
1. Konsep: Boxplot adalah visualisasi grafis yang digunakan untuk menganalisis distribusi data dan mengidentifikasi adanya outlier. Boxplot menampilkan beberapa ukuran statistik, termasuk Q1, Q3, median, serta batas atas dan batas bawah untuk outlier.
2. Cara Kerja:
   - Boxplot terdiri dari sebuah kotak (box) yang menunjukkan rentang IQR (dari Q1 hingga Q3).
   - Garis di dalam kotak menunjukkan posisi median.
   - Whisker atau garis lurus yang terhubung dengan kotak menunjukkan rentang data yang dianggap tidak sebagai outlier.
   - Titik-titik di luar whisker menunjukkan adanya outlier.
  
![outliers_boxplot](https://github.com/nikofebrianur/Machine-Learning-Terapan/assets/42314371/994b7ef1-fe6d-4bc4-94a5-5800322fe11f)
###### Gambar 6.1 Visualisasi outliers menggunakan boxplot

Dengan menggunakan metode IQR (Interquartile Range) dan visualisasi boxplot, kita dapat mengidentifikasi outlier dalam data dengan lebih akurat. Outlier adalah data yang jauh dari nilai-nilai lain dalam dataset dan bisa memberikan pengaruh yang tidak diinginkan pada model analisis dan prediksi.

Langkah-langkah Menggunakan IQR untuk Mengidentifikasi Outlier:
Hitung IQR:

IQR adalah selisih antara Q3 (Kuartil Ketiga) dan Q1 (Kuartil Pertama), yaitu 25% data teratas dan 25% data terbawah.
Rumusnya:
𝐼
𝑄
𝑅
=
𝑄
3
−
𝑄
1
IQR=Q3−Q1
Tentukan Batas Atas dan Bawah untuk Outlier:

Batas Bawah: Q1 - 1.5 * IQR
Batas Atas: Q3 + 1.5 * IQR
Data yang berada di luar batas bawah atau atas ini dianggap sebagai outlier.
Visualisasi Boxplot:

Boxplot adalah grafik yang menunjukkan distribusi data berdasarkan kuartil. Ini memudahkan kita untuk mengidentifikasi outlier yang ditampilkan sebagai titik di luar batas-batas whisker dari boxplot.
Pada boxplot, kita dapat dengan cepat melihat distribusi, median, dan juga apakah ada pencilan yang perlu diperhatikan.
Menangani Outlier:
Setelah mengidentifikasi outlier, kita dapat memilih salah satu dari beberapa tindakan:

Menghapus Outlier: Jika outlier merupakan data yang keliru atau tidak sesuai dengan konteks analisis, kita dapat menghapusnya.
Mengelola Outlier: Jika outlier penting untuk analisis (misalnya pada kasus medis atau keuangan), kita dapat mempertimbangkan untuk meredam pengaruhnya dengan teknik seperti robust scaling atau mengganti nilai outlier dengan rata-rata atau median.
Analisis Sebaran Dataset dengan Histogram:
Histogram adalah alat visualisasi yang digunakan untuk melihat distribusi frekuensi dari suatu variabel.
Histogram menampilkan seberapa sering nilai tertentu muncul dalam data, dengan nilai variabel di sepanjang sumbu x dan frekuensi munculnya nilai tersebut di sumbu y.
Interval/Bin: Histogram dibagi menjadi interval atau bin, dan tinggi setiap bin menunjukkan jumlah data yang jatuh ke dalam rentang tersebut.
Histogram berguna untuk:
Melihat apakah distribusi data bersifat normal, miring ke kiri/kanan, atau multimodal.
Memahami pola distribusi secara keseluruhan, termasuk potensi keberadaan outlier.
Dengan menggunakan IQR, boxplot, dan histogram, kita dapat mengidentifikasi outlier dan memahami distribusi dataset secara lebih baik, memastikan bahwa analisis statistik dan model prediksi lebih akurat serta tidak terganggu oleh data yang ekstrem.

Berikut adalah langkah-langkah untuk menganalisa sebaran dataset menggunakan histogram:

1. Membagi Data Menjadi Interval atau Bin:
   - Pertama, kita perlu membagi data ke dalam interval atau bin. Jumlah dan lebar interval dapat bervariasi tergantung pada dataset dan tujuan analisis.
   - Untuk mendapatkan jumlah interval yang tepat, kita bisa menggunakan aturan umum seperti aturan Sturges atau Scott's normal reference rule.

2. Menghitung Frekuensi:
   - Selanjutnya, kita menghitung frekuensi munculnya data di setiap interval. Frekuensi dapat dihitung sebagai jumlah observasi yang jatuh di dalam setiap interval.

3. Visualisasi dengan Histogram:
   - Dengan menggunakan data frekuensi yang telah dihitung, kita dapat membuat histogram.
   - Pada sumbu horizontal, kita menempatkan interval atau bin, sedangkan pada sumbu vertikal, kita menampilkan frekuensi munculnya data dalam interval tersebut.
   - Untuk menggambarkan histogram, kita dapat menggunakan bar-chart dengan lebar bar yang mencerminkan lebar interval.

4. Interpretasi:
   - Dengan melihat histogram, kita dapat menganalisa sebaran dataset secara visual.
   - Kita dapat melihat apakah data memiliki distribusi normal, simetris, asimetris (ke kiri atau ke kanan), atau memiliki pola tertentu seperti bimodal (dua puncak).
   - Kita juga dapat melihat kisaran nilai yang paling sering muncul dan sebaran nilai di dalam dataset.

Histogram membantu kita memahami pola dan sebaran data dengan cepat. Analisis sebaran dataset menggunakan histogram memungkinkan kita mengidentifikasi tipe distribusi data, menentukan apakah terdapat outlier, dan memperoleh gambaran umum tentang data tersebut. 

Hal ini dapat menjadi langkah awal dalam eksplorasi data sebelum melakukan analisis lebih lanjut atau membangun model prediksi.

![histogram](https://github.com/nikofebrianur/Machine-Learning-Terapan/assets/42314371/88d5ec99-beca-4d6f-a34d-a95921c4fe79)
###### Gambar 6.2 Sebaran dataset

![pairplot](https://github.com/nikofebrianur/Machine-Learning-Terapan/assets/42314371/ab84ebd0-4fe9-4436-9439-359c83ee7eb3)
###### Gambar 6.2 Korelasi PremiumPrice dengan fitur lainnya

Terakhir, untuk menganalisa keterkaitan antara fitur numerik dan fitur kategori, kita dapat menggunakan correlation matrix dengan fungsi pandas dan visualisasi heatmap menggunakan library seaborn. 

Heatmap menunjukkan tingkat korelasi antara setiap pasangan fitur numerik dan fitur kategori. Warna dalam heatmap mencerminkan tingkat korelasi, di mana warna lebih terang menunjukkan korelasi yang lebih kuat, sedangkan warna lebih gelap menunjukkan korelasi yang lebih lemah atau tidak ada korelasi.

Kita dapat melihat hubungan positif atau negatif antara fitur numerik dan fitur kategori berdasarkan nilai korelasi. Analisis ini membantu dalam memahami keterkaitan antar fitur-fitur dalam dataset dan dapat memberikan wawasan yang berguna untuk pemilihan fitur, pemodelan, atau analisis lebih lanjut.

Dengan menggunakan correlation matrix dan visualisasi heatmap, kita dapat dengan mudah menganalisa keterkaitan antara fitur numerik dan fitur kategori dalam dataset secara visual dan kuantitatif.

![correlation_matrix](https://github.com/nikofebrianur/Machine-Learning-Terapan/assets/42314371/9b8346a0-2cab-4184-be8a-789a447e7f6d)
###### Gambar 6.3 Matriks korelasi PremiumPrice 

Dapat dilihat dari gambar matriks di atas bahwa variabel *AnyTransplants*, *AnyChronicDiseases*, dan *NumberOfMajorSurgeries* memiliki warna heatmap yang terang dan ini menunjukkan bahwa ketiga variabel tersebut memiliki korelasi dengan variabel *PremiumPrice*     

## 7. Data Preparation

Berikut ada teknik yang digunakan dalam proses data preparation, yaitu:

Proses *One Hot Encoding* pada fitur kategorikal adalah teknik yang digunakan untuk mengubah variabel kategorikal menjadi representasi numerik yang dapat digunakan dalam model *machine learning*. 

Hal ini diperlukan karena sebagian besar algoritma *machine learning* hanya dapat bekerja dengan input numerik.

Pandas *library* menyediakan fungsi pd.get_dummies() yang memudahkan dalam melakukan *One Hot Encoding*. Fungsi ini akan menghasilkan kolom-kolom baru yang mewakili setiap nilai unik dari fitur kategorikal. Jika suatu baris memiliki nilai tersebut, kolom yang sesuai akan diatur menjadi 1, sedangkan kolom lainnya akan menjadi 0.

Misalnya, jika terdapat fitur "Warna" dengan nilai "Merah", "Biru", dan "Hijau", setelah One Hot Encoding akan terbentuk tiga kolom baru: "Warna_Merah", "Warna_Biru", dan "Warna_Hijau". 

Jika suatu baris memiliki nilai "Merah" pada fitur "Warna", maka kolom "Warna_Merah" akan diatur menjadi 1, sedangkan kolom lainnya akan menjadi 0.

Proses pembagian dataset menjadi data training dan data testing penting dalam pengembangan model *machine learning*. Ini dilakukan untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya dan untuk menghindari *overfitting*. 

Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk menguji seberapa baik model yang dilatih dapat melakukan prediksi pada data yang belum pernah dilihat sebelumnya. 

Dengan memisahkan data training dan data testing, kita dapat mengukur sejauh mana model dapat mengeneralisasi dan memprediksi dengan akurat pada data baru.

Rasio 80:20 sering digunakan sebagai perbandingan pembagian data training dan data testing. Data training sebesar 80% digunakan untuk melatih model, sementara data testing sebesar 20% digunakan untuk menguji performa model. 

Rasio ini merupakan aturan praktis umum yang memberikan keseimbangan antara memiliki jumlah data yang cukup untuk melatih model dan menyediakan data yang cukup untuk menguji performa model. 

Namun, rasio ini dapat bervariasi tergantung pada karakteristik dataset dan kebutuhan proyek tertentu.

## 8. Modeling
Dalam proyek ini, penggunaan algoritma SVR (Support Vector Regression) dan Huber Regressor diharapkan dapat memberikan solusi yang efektif untuk memprediksi premi asuransi kesehatan. Kedua algoritma ini dipilih karena kekuatannya dalam menangani data regresi, terutama yang rentan terhadap outliers atau noise.

Berikut adalah penjelasan lebih rinci tentang keduanya:

1. Support Vector Regression (SVR):
Konsep: SVR merupakan adaptasi dari Support Vector Machines (SVM) yang digunakan untuk tugas regresi. Fokus SVR adalah memprediksi variabel target dengan meminimalkan kesalahan prediksi, sambil memaksimalkan margin yang diizinkan. Algoritma ini berusaha untuk menemukan sebuah hyperplane yang memprediksi nilai target dengan toleransi tertentu.
Cara Kerja: Dalam SVR, model akan mencoba untuk memasukkan sebanyak mungkin titik data ke dalam margin, dan hanya mempertimbangkan titik data yang melampaui margin tersebut (disebut sebagai support vectors). Fungsi kernel dalam SVR memberikan fleksibilitas untuk menangani hubungan non-linear antara fitur-fitur input dan output dengan cara mentransformasikan data ke dimensi yang lebih tinggi.
Keuntungan:
SVR sangat fleksibel dalam memilih kernel (linear, polynomial, RBF) untuk mengatasi masalah non-linear.
Algoritma ini bekerja baik dengan data yang memiliki outliers moderat karena hanya memperhitungkan support vectors yang berada di luar margin.
SVR dapat memberikan model yang kuat untuk memprediksi hubungan kompleks dengan menggunakan kernel yang sesuai.
2. Huber Regressor:
Konsep: Huber Regressor adalah algoritma regresi robust yang menggabungkan kelebihan metode Least Squares dan Least Absolute Deviations. Fungsi kerugian Huber linear terhadap outliers yang signifikan, sehingga membuat model ini lebih tahan terhadap outliers daripada metode Least Squares konvensional.
Cara Kerja: Huber Regressor menghitung residual antara prediksi dan nilai sebenarnya. Jika residual lebih kecil dari nilai ambang batas tertentu (threshold), Huber menggunakan metode Least Squares, tetapi jika residual lebih besar dari threshold, algoritma menggunakan metode Least Absolute Deviations untuk meminimalkan efek outliers. Dengan cara ini, Huber Regressor mampu menyesuaikan dengan data yang memiliki outliers besar tanpa terganggu oleh outliers tersebut.
Keuntungan:
Huber Regressor sangat baik dalam mengatasi dataset yang memiliki outliers yang signifikan.
Algoritma ini menjaga keseimbangan antara ketahanan terhadap outliers dan presisi pada data yang tidak mengandung outliers.
Algoritma ini dapat memberikan performa yang stabil dalam kondisi data yang bervariasi.
Keuntungan Menggabungkan Kedua Algoritma:
SVR memberikan solusi yang baik untuk memodelkan data dengan hubungan non-linear dan noise moderat, sementara Huber Regressor sangat efektif dalam menangani outliers ekstrem tanpa kehilangan ketepatan prediksi pada data yang tidak mengandung outliers.
Penggunaan kedua algoritma ini memberikan fleksibilitas yang lebih besar dalam menghadapi karakteristik data yang berbeda. Dengan mengeksplorasi kekuatan masing-masing algoritma, proyek ini dapat menghasilkan prediksi yang lebih akurat dan robust terhadap variasi dan anomali dalam dataset.
Evaluasi dan Optimalisasi:
Dengan menggunakan PyCaret, model SVR dan Huber Regressor dapat dioptimalkan lebih lanjut melalui penyesuaian hyperparameter dan perbandingan kinerja dengan algoritma regresi lainnya.
PyCaret memungkinkan proses tuning hyperparameter seperti kernel type untuk SVR atau alpha dan epsilon pada Huber Regressor sehingga hasil prediksi premi asuransi bisa lebih akurat.
Setelah model dilatih, evaluasi akan dilakukan menggunakan metrik Mean Squared Error (MSE) yang memberikan indikasi seberapa baik prediksi model mendekati nilai sebenarnya.
Kesimpulan:
Menggunakan kombinasi algoritma SVR dan Huber Regressor untuk prediksi premi asuransi kesehatan adalah pendekatan yang sangat sesuai. Kedua algoritma ini dipilih karena kemampuannya menangani data yang mengandung noise dan outliers, serta fleksibilitas dalam memodelkan hubungan yang rumit antara variabel risiko dan premi. Pendekatan ini diharapkan mampu menghasilkan model prediksi yang akurat dan robust, memastikan hasil yang lebih dapat diandalkan untuk keputusan bisnis di industri asuransi.

### Tahapan yang dilakukan
Berikut adalah urutan tahapan yang dilakukan dalam proses modeling:
 - Melatih model dengan data training dengan menggunakan algoritma *Huber Regressor* dan *SVR*
 - Dalam tahap training, pengujian model dilakukan dengan menggunakan parameter default bawaan *library*
 - Melakukan pengujian dengan data training
 - Melakukan pengujian dengan data testing
 - Pengukuran menggunakan metriks *MSE*,MAE,R*MSE* dan R2 dengan menggunakan lirbary sklearn. 
 - Melihat hasil performa model antara hasil data training dan data testing
 - Meningkatkan performa model dengan menerapkan grid search atau *hyperparameter* pada model
 - Menggunakan *hyperparameter* pada *Huber Regressor* yaitu param_grid = { 'epsilon': [1.0, 1.5, 2.0],'alpha': [0.0001, 0.001, 0.01], 'max_iter': [100, 200, 300]}
 - Menggunakan *hyperparameter* pada *SVR* yaitu param_grid = {'kernel': ['linear', 'rbf'],'C': [0.1, 1, 10],'epsilon': [0.1, 0.2, 0.3]}
 - Setelah pengujian *hyperparameter*, *Huber Regressor* mendapatkan param terbaik yaitu: {'alpha': 0.01, 'epsilon': 2.0, 'max_iter': 100}
 - Setelah pengujian *hyperparameter*, *SVR* mendapatkan param terbaik yaitu: {'C': 1, 'epsilon': 0.1, 'kernel': 'linear'

 - Huber_*MSE*: Rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual pada data training adalah sekitar 18,480,694.56826, sedangkan pada data testing sekitar 25,177,990.875244. Hasil ini menunjukkan bahwa model memiliki tingkat kesalahan yang sedikit lebih tinggi pada data testing dibandingkan dengan data training.
 - *SVR*_*MSE*: Rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual pada data training adalah sekitar 40,114,366.684487, sedangkan pada data testing sekitar 42,742,505.675315. Hasil ini menunjukkan bahwa model testing memiliki tingkat kesalahan yang sedikit lebih tinggi pada data testing dibandingkan dengan data training.

Kesimpulan yang dapat diambil dari paparan di atas ialah adanya overfitting pada data training karena model cenderung memiliki tingkat kesalahan yang lebih tinggi pada data testing. 

### Kelebihan dan Kekurangan
 - Algoritma *Huber Regressor* memiliki keunggulan lebih tahan terhadap adanya *outlier* dan oleh karena itu dapat memberikan hasil yang lebih konsisten dan stabil dibandingkan dengan regresi linier biasa
 - Algoritma *Huber Regressor* memiliki kekurangan dalam pemilihan nilai yang tepat untuk delta dapat mempengaruhi kinerja algoritma, dan penyesuaian parameter ini sering kali memerlukan pengujian dan penyesuaian manual. 
 - Algoritma *SVR* memiliki keunggulan dapat menggunakan fungsi kernel yang berbeda untuk mengubah data input menjadi ruang fitur yang lebih tinggi, memungkinkan pemodelan yang lebih fleksibel dan penanganan kasus-kasus di mana hubungan antara fitur dan target bersifat non-linear.
 - Algoritma *SVR* memiliki kekurangan sangat sensitif terhadap skala data input. Oleh karena itu, penting untuk melakukan normalisasi atau penskalaan data sebelum menggunakan *SVR* untuk menghindari bias yang tidak diinginkan dalam model.

Berdasarkan pertimbangan kelebihan dan kekurangan di atas, maka algoritma *Huber Regressor* dinilai lebih unggul dari *SVR* dan setelahnya perlu menerapkan *hyperparameter* saat model masuk ke evaluasi.

## 9. Evaluation

Metrik evaluasi yang digunakan dalam proyek ini ialah sebagai berikut:

### *MSE*
Mean Square Error (*MSE*) adalah salah satu metrik evaluasi yang digunakan untuk mengukur sejauh mana perbedaan antara nilai prediksi dan nilai sebenarnya dalam masalah regresi. 

Model evaluasi *MSE* menghitung rata-rata dari kuadrat selisih antara nilai prediksi dan nilai sebenarnya. Semakin kecil *MSE*, semakin baik model tersebut dalam melakukan prediksi yang akurat.

Berikut adalah langkah-langkah untuk menghitung *MSE*:

1. Mulai dengan memiliki kumpulan data yang terdiri dari pasangan nilai sebenarnya (y) dan nilai prediksi (ŷ) untuk sejumlah contoh atau sampel.

2. Hitung selisih antara nilai sebenarnya dan nilai prediksi untuk setiap contoh. Selisih ini merupakan error atau kesalahan prediksi untuk masing-masing contoh.

3. Kuadratkan setiap selisih. Ini dilakukan untuk memastikan bahwa setiap error memiliki kontribusi positif terhadap nilai *MSE*, tanpa mempertimbangkan apakah prediksi lebih rendah atau lebih tinggi dari nilai sebenarnya.

4. Hitung rata-rata dari kuadrat selisih. Caranya adalah dengan menjumlahkan semua kuadrat selisih dan membaginya dengan jumlah contoh.

   *MSE* = (Σ (y - ŷ)²) / n
   
   di mana:
   - Σ menunjukkan penjumlahan
   - y adalah nilai sebenarnya
   - ŷ adalah nilai prediksi
   - n adalah jumlah contoh atau sampel dalam dataset.

5. Setelah menghitung *MSE*, semakin kecil nilai *MSE*, semakin baik model dalam melakukan prediksi yang akurat. *MSE* memiliki satuan yang berbeda dengan variabel yang dievaluasi, karena hasilnya berupa kuadrat. Oleh karena itu, *MSE* seringkali diinterpretasikan dalam konteks yang lebih luas, atau perbandingannya dibandingkan dengan metrik evaluasi lainnya.

Tabel 9.1 Tabel hasil running evaluasi model setelah menggunakan *hyperparameter*
|           | Huber         | SVR            |
|-----------|---------------|----------------|
| train_MSE | 18480694.56826 | 40114366.684487 |
| test_MSE  | 25177990.875244 | 42742505.675315 |
| eval_train | 18104561.09466 | 18614533.696279 |
| eval_test  | 24212812.866265 | 25605704.542962 |

Berikut adalah grafik hasil evaluasi model setelah dilakukan penerapan *hyperparameter*.

![model_eval_hy_params](https://github.com/nikofebrianur/Machine-Learning-Terapan/assets/42314371/d374d1d7-fcc2-4afb-9745-a221cacb2453)
###### Gambar 9.1 Hasil evaluasi model setelah penerapan *hyperparameter*

*MSE* merupakan metrik evaluasi yang umum digunakan dalam masalah regresi karena memperhitungkan perbedaan antara nilai prediksi dan nilai sebenarnya secara keseluruhan dan memberikan bobot yang lebih besar pada perbedaan yang besar. 

Namun, *MSE* juga memiliki kelemahan yaitu sensitif terhadap *outlier*, artinya nilai ekstrem yang sangat berbeda dapat mempengaruhi *MSE* secara signifikan. 

Oleh karena itu, terkadang metrik evaluasi alternatif seperti *Mean Absolute Error* (MAE) juga digunakan untuk memberikan gambaran yang lebih lengkap tentang kinerja model.

Berdasarkan hasil evaluasi model setelah menggunakan *hyperparameter*, kita dapat mengambil beberapa kesimpulan:

1. *Mean Squared Error* (*MSE*) - Train Set:
   - Model *Huber* memiliki *MSE* train set sebesar 18,480,694.56826.
   - Model *SVR* memiliki *MSE* train set sebesar 40,114,366.684487.
   - Model *Huber* memiliki nilai *MSE* yang lebih rendah dibandingkan dengan model *SVR* pada data training. Artinya, model Huber mampu melakukan prediksi yang lebih akurat daripada model *SVR*.

2. *Mean Squared Error* (*MSE*) - Test Set:
   - Model *Huber* memiliki *MSE* test set sebesar 25,177,990.875244.
   - Model *SVR* memiliki *MSE* test set sebesar 42,742,505.675315.
   - Model *Huber* juga memiliki nilai *MSE* yang lebih rendah dibandingkan dengan model *SVR* pada data testing. Hal ini menunjukkan bahwa model *Huber* lebih baik dalam melakukan prediksi pada data yang belum pernah dilihat sebelumnya.

3. Evaluation Score (eval) - Train Set:
   - Model *Huber* memiliki evaluasi score train set sebesar 18,104,561.09466.
   - Model *SVR* memiliki evaluasi score train set sebesar 18,614,533.696279.
   - Meskipun model *Huber* memiliki *MSE* yang lebih rendah pada data training, evaluasi score keduanya memiliki perbedaan yang cukup kecil. Kedua model memberikan performa yang serupa pada data training.

4. Evaluation Score (eval) - Test Set:
   - Model *Huber* memiliki evaluasi score test set sebesar 24,212,812.866265.
   - Model *SVR* memiliki evaluasi score test set sebesar 25,605,704.542962.
   - Model *Huber* juga memberikan evaluasi score yang lebih rendah dibandingkan dengan model *SVR* pada data testing. Hal ini menunjukkan bahwa model *Huber* lebih baik dalam melakukan prediksi yang lebih akurat pada data yang belum pernah dilihat sebelumnya.

Berdasarkan kesimpulan di atas, model *Huber* memiliki performa yang lebih baik daripada model *SVR* dalam memprediksi premi asuransi kesehatan. Model *Huber* memiliki *MSE* yang lebih rendah baik pada data training maupun test set, serta memberikan evaluasi score yang lebih baik pada data testing. 

Oleh karena itu, model *Huber* dapat dianggap lebih optimal dalam proyek ini untuk memperkirakan premi asuransi kesehatan dengan tingkat akurasi yang lebih tinggi.

## 10. Kesimpulan 

Proyek ini berhasil membuktikan bahwa model Huber Regressor memiliki keunggulan dalam memprediksi premi asuransi kesehatan dengan akurasi yang lebih tinggi dibandingkan SVR. Evaluasi menunjukkan bahwa model Huber menghasilkan Mean Squared Error (MSE) yang lebih rendah di kedua set data (training dan testing), menunjukkan prediksi yang lebih presisi serta performa yang lebih stabil. Dengan hasil ini, model Huber dapat memberikan berbagai manfaat, baik bagi perusahaan asuransi maupun calon pemegang polis.

Manfaat bagi Perusahaan Asuransi:
Penentuan Premi yang Lebih Akurat:

Dengan menggunakan model Huber, perusahaan dapat menetapkan premi yang lebih akurat berdasarkan faktor risiko individu, mengurangi ketidakpastian dan meminimalkan risiko kerugian finansial akibat salah perhitungan.
Akurasi yang lebih tinggi juga membantu dalam pengelolaan risiko, karena premi yang ditetapkan lebih sesuai dengan profil risiko masing-masing pemegang polis.
Transparansi yang Ditingkatkan:

Model ini memberikan kemampuan bagi perusahaan untuk menjelaskan dengan lebih detail faktor-faktor yang mempengaruhi premi. Pemahaman yang lebih baik dari pihak perusahaan meningkatkan transparansi, yang pada akhirnya dapat meningkatkan kepercayaan antara perusahaan dan nasabah.
Manfaat bagi Calon Pemegang Polis:
Premi yang Adil dan Akurat:

Calon pemegang polis akan mendapatkan premi yang lebih adil karena penetapan didasarkan pada perhitungan yang lebih tepat dan mempertimbangkan faktor risiko yang relevan, sehingga mereka hanya membayar premi yang sesuai dengan risiko yang ditanggung.
Pemahaman yang Lebih Baik tentang Premi:

Dengan adanya transparansi yang lebih baik, calon pemegang polis dapat lebih memahami alasan di balik penetapan premi mereka, memberikan rasa keadilan yang lebih besar dan meningkatkan pengalaman mereka sebagai nasabah.
Langkah Tindak Lanjut:
Implementasi Model:

Perusahaan asuransi dapat langsung menerapkan model Huber yang dikembangkan ini ke dalam sistem penetapan premi mereka. Ini akan memberikan manfaat langsung dalam penentuan premi yang lebih tepat.
Peningkatan Data dan Pemeliharaan Model:

Perusahaan perlu terus memperkaya dataset mereka dengan data baru agar model tetap relevan dan akurat. Model Huber perlu dipelihara secara berkala, baik dari segi pelatihan ulang model maupun pembaruan data.
Evaluasi dan Peningkatan:

Seiring berjalannya waktu, perusahaan dapat terus mengevaluasi performa model di dunia nyata dan melakukan optimasi lebih lanjut untuk meningkatkan akurasi dan kemampuan model, serta menyesuaikannya dengan perubahan tren risiko yang terjadi.
Kesimpulan:
Dengan mengimplementasikan model Huber Regressor, perusahaan asuransi dapat meningkatkan akurasi penetapan premi asuransi kesehatan, memberikan manfaat yang nyata bagi calon pemegang polis, serta memperkuat transparansi dan efisiensi dalam operasi bisnis mereka. Evaluasi dan perbaikan berkelanjutan pada model akan memastikan keberlanjutan manfaat ini di masa mendatang.

## References: 
Chauluka M, Uzochukwu B, and Chinkhumba J, "Factors Associated With Coverage of Health Insurance Among Women in Malawi," *Frontiers in Health Services*, vol.2, 2022.

Michael Chernew, David M Cutler, and Patricia Seliger Keenan, "Increasing Health Insurance Costs and the Decline in Insurance Coverage,
" *Health Services Research*, vol.40, no.10.1111. 2005. 

Samantha Artiga, Petry Ubri, and Julia Zur, "The Effects of Premiums and Cost Sharing on Low-Income Populations: Updated Review of Research Findings The Effects of Premiums and Cost Sharing on Low-Income Populations 2," *Issue Brief*, 2017



