import streamlit as st
from streamlit_lottie import st_lottie

logo = "assets/logo.png"
lottie_robot = "https://lottie.host/9a012d01-f820-44f9-b293-777c492451a9/Hw5C5jR7Gi.json"
lottie_scan = "https://lottie.host/16a968a3-7b0e-4794-9ceb-79d0e7c0d508/zActw4SWyv.json"
lottie_comment = "https://lottie.host/ed7a3fa2-d4a8-46ea-ab27-25e7e197c771/6wA7wuJoyx.json"
lottie_attack = "https://lottie.host/4c89a4d2-af42-4ac5-9540-ee6d7fc549a2/ecpDGaRLGP.json"
lottie_email = "https://lottie.host/7724d991-ac93-489b-bffa-3146aa89ed9c/Rpe6VVsUwQ.json"

st.set_page_config(
    page_title="Captcha Detector",
    page_icon="assets/logo.png",
    layout="centered"
)

_left, mid, _right = st.columns([0.2,0.5,0.2])
with mid:
    st_lottie(lottie_robot, key = "robot")
        
st.markdown(
    "<h3 style='text-align: center;'> </h3>", unsafe_allow_html=True)
st.markdown(
    "<h1 style='text-align: center;'>Captcha Code Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center;'>Image(hCaptcha) dan Text(Captcha) </h3>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center;'>Apa itu Captcha ?</h3>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>Captcha adalah fitur yang memberikan lapisan keamanan pada website untuk memastikan bahwa website diakses oleh manusia sungguhan, bukan robot. Contoh captcha yaitu ketika Anda membuka suatu website, Anda diminta untuk mengisi rangkaian huruf, gambar, atau puzzle gambar lebih dulu.</h6>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>Pada dasarnya, CAPTCHA adalah singkatan dari Completely Automated Public Turing Test to Tell Computers and Humans Apart.Dari istilah bahasa Inggris ini, bisa disimpulkan bahwa arti captcha yaitu proses otomatis yang dilakukan untuk membedakan apakah pengunjung website merupakan seorang manusia atau robot (Computers and Humans Apart).</h6>", unsafe_allow_html=True)

_left, mid, _right = st.columns([0.2,0.5,0.2])
with mid:
    st_lottie(lottie_scan, key = "scan")

st.markdown(
    "<h6 style='text-align: justify;'>Akses autentikasinya berupa validasi dengan desain khusus, yang memastikan bahwa pengunjung yang mengakses website bukanlah robot. Oleh karena itu, pertanyaan autentikasi selalu dibuat sedemikian rupa agar yang bisa memecahkannya hanya manusia sungguhan.</h6>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center;'> </h3>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center;'>Fungsi Captcha Bagi Website ?</h3>", unsafe_allow_html=True)  
st.markdown(
    "<h4 style='text-align: justify;'>1. Mencegah Komentar Spam</h4>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>Pernahkah Anda diminta memasukkan captcha dari kombinasi angka, huruf, atau gambar lebih dulu ketika ingin menuliskan komentar di blog atau website? Nah, hal ini diterapkan untuk mencegah komentar spam di website atau blog.</h6>", unsafe_allow_html=True)

_left, mid, _right = st.columns([0.2,0.2,0.2])
with mid:
    st_lottie(lottie_comment, key = "comment")

st.markdown(
    "<h4 style='text-align: justify;'>2. Menghalangi Dictionary Attack pada Website atau Blog</h4>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>Dikutip dari Wikipedia, dictionary attack adalah serangan untuk membobol sistem keamanan dengan memecahkan kode pada mekanisme cipher atau autentikasi, menggunakan kata atau frasa yang terdapat pada kamus (dictionary) sebanyak ribuan atau bahkan jutaan kali percobaan. Dictionary attack pada website atau blog ini bisa sampai pada kasus pencurian akun, loh. Dengan adanya captcha, Anda bisa mencegah serangan jenis ini.</h6>", unsafe_allow_html=True)

_left, mid, _right = st.columns([0.2,0.3,0.2])
with mid:
    st_lottie(lottie_attack, key = "attack")

st.markdown(
    "<h4 style='text-align: justify;'>3. Melindungi Alamat Email dari Scraper</h4>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>Scraper adalah salah satu ancaman keamanan yang membahayakan email, berupa bot yang bisa ‘mencuri’ dan mengekspor alamat email dari sebuah website, dan bisa berujung pada pengambilalihan akses. Captcha di sini akan berfungsi melindungi dan menyembunyikan alamat email Anda dari scraper sehingga para spammer tidak bisa sembarangan menyebarkan spam dan mencari alamat email Anda.</h6>", unsafe_allow_html=True)

_left, mid, _right = st.columns([0.2,0.2,0.2])
with mid:
    st_lottie(lottie_email, key = "email")

st.markdown(
    "<h3 style='text-align: center;'>Jenis Captcha</h3>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: justify;'>1. Text Captcha</h4>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>Captcha teks merupakan yang paling sering ditemui dan umum digunakan. Selain itu, jenis ini merupakan yang paling pertama digunakan sejak keberadaan captcha untuk membedakan antara manusia dengan robot. Biasanya, captcha teks terdiri dari susunan kombinasi angka, huruf, atau simbol. Pengunjung akan diminta mengisi kode dari karakter-karakter tersebut yang tatanannya ditampilkan secara acak.</h6>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: justify;'>2. Picture Recognition Captcha</h4>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>Captcha jenis ini menggunakan gambar untuk proses autentikasinya. Captcha gambar merupakan versi yang dikembangkan dari jenis teks, dan memang ditujukan untuk menggantikan captcha text. Untuk jenis picture recognition ini, pengunjung harus memilih bagian foto yang sudah ditentukan, biasanya dengan instruksi untuk mengklik foto yang sesuai. Gambar yang ditampilkan pun bermacam-macam, biasanya berupa elemen grafis, pemandangan, rambu lalu lintas, dan sebagainya. Picture recognition captcha sedikit menjadi masalah bagi pengunjung website yang memiliki gangguan penglihatan karena harus benar-benar teliti melihatnya.</h6>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: justify;'>3. jQuery Slider Captcha</h4>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>jQuery Slider Captcha adalah metode autentikasi yang berbentuk slider, mengharuskan pengunjung website menggeser sebuah elemen di layar menggunakan mouse untuk menyelesaikannya. Biasanya, pengguna hanya perlu menggeser kotak dari kiri ke kanan untuk melakukan verifikasi keamanan. enis ini ada yang berbentuk puzzle yang harus digeser agar pas di area yang benar, atau ada juga yang hanya berupa perintah untuk menggeser elemen tertentu.</h6>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: justify;'>4. Audio Captcha</h4>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>Captcha audio digunakan sebagai alternatif untuk pengunjung website yang memiliki gangguan penglihatan atau penyandang tuna netra. Sesuai namanya, jenis ini menggunakan suara untuk mendikte kode yang perlu ditulis untuk autentikasi. Audio captcha juga terkadang disertakan otomatis pada captcha teks, dengan ikon speaker yang bisa diklik untuk membacakan kode di layar.</h6>", unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align: center;'>Kesimpulan</h3>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: justify;'>Selesai! Di artikel ini, Anda sudah mempelajari tentang apa itu captcha, arti captcha, fungsinya, serta jenis-jenisnya. Setelah membaca panduan kami, semoga Anda sebagai pengunjung maupun pemilik website tidak lagi bingung dan kesal ketika menjumpai captcha. Meski terkadang sedikit mengganggu, captcha memiliki peran yang sangat penting bagi keamanan website atau blog. Kegunaan captcha di antaranya adalah untuk mencegah registrasi akun palsu, spam, dictionary attack, pencurian email, dan mengamankan transaksi online.</h6>", unsafe_allow_html=True)