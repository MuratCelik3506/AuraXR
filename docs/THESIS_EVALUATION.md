# Tez Değerlendirmesi — Master Tezi Jüri Perspektifi

**Değerlendiren:** Claude Sonnet 4.6 (Yapay Zeka Jüri Simülasyonu)  
**Tarih:** 2026-07-01  
**Konu:** AuraXR — VR/XR Ortamında Gerçek Zamanlı Temporal El Kavrama Üretimi

---

## Genel Puan: 5.5 / 10

> Bu tez, teknolojik olarak hırslı bir probleme iyi düşünülmüş bir mimari yaklaşım sunmaktadır. Ancak **bilimsel kanıtlama açığı** ciddi boyuttadır: altı ana araştırma sorusundan beşi sayısal olarak yanıtsız kalmış, en kritik metrik (contact ratio) hedefin çok altında seyretmiş, Unity fizik değerlendirmesi tamamlanmamıştır. Mevcut haliyle tez bir "çalışan sistem + dürüst başarısızlık kaydı" olarak değerlendirilebilir; ancak bir **akademik katkı kanıtı** olarak henüz yeterli değildir.

---

## 1. Problem Tanımı ve Motivasyon — 7 / 10

**Güçlü yönler:**
- VR/XR'da controller-driven grasp problemi net tanımlanmış; "canned animasyon vs. gerçek zamanlı AI" ayrımı ikna edici.
- Object-relative koordinat çerçevesi seçimi (§3.3) metodolojik olarak doğru ve savunulabilir.
- OakInk (statik) + HOT3D (temporal) veri setlerini tek mimariye dahil etme fikri özgün.

**Zayıf yönler:**
- Giriş bölümü (§1) madde listesi düzeyinde kalmış; "neden bu problem önemli" sorusuna akademik literatürle desteklenmiş kapsamlı bir yanıt yok.
- Katkılar bölümünde öz-eleştiri (**4–6 numaralı katkıların henüz kanıtlanmadığı**) doğruca yazılmış — bu dürüstlük değerli, ama aynı zamanda tezin olgunlaşmamışlığını da sergiliyor.

---

## 2. İlgili Çalışmalar — 5 / 10

**Güçlü yönler:**
- GrabNet, ContactOpt, GraspTTA, PointNet ailesi ve temporal modeller belirtilmiş.
- CVAE ve diffusion tabanlı karşılaştırma notedilmiş.

**Zayıf yönler:**
- İlgili çalışmalar bölümü (§2) tamamen madde listesi formatında; hiçbir çalışmayla sayısal karşılaştırma yapılmamış.
- "Bu tez GrabNet'ten nasıl farklılaşıyor?" sorusu metinde doğrudan cevaplanmıyor.
- Diffusion tabanlı grasp üretimi (DiffusionGrasp vb.) "future work" olarak not edilmiş ama bu alanın neden tercih edilmediği açıklanmamış.
- **Kritik eksik:** Mevcut SOTA ile sayısal karşılaştırma tablosu yok. OakInk üzerinde GrabNet gibi bir baseline ile 9.7° geodesic değerinin ne anlama geldiği bilinmiyor.

---

## 3. Yöntem — 7.5 / 10

**Güçlü yönler:**
- Mimari (Mini PointNet + FiLM + GRU + JointSelfAttention + CVAE) mantıklı bileşenlerden oluşuyor; her birinin rolü §3'te açıkça tanımlanmış.
- FiLM conditioning'in object geometry'yi temporal encoder'a enjekte etme biçimi doğru.
- Multi-task kayıp fonksiyonu (§3.6) kapsamlı; vel/acc regularizasyonu ve fingertip position loss `L_tip` eklemeleri mantıklı.
- ONNX export süreci ve Unity InferenceEngine kısıtlarıyla başa çıkma (§12.8) gerçek mühendislik sorununu çözme kapasitesi gösteriyor.

**Zayıf yönler:**
- **CVAE latent boyutu (64) ve KL ağırlığı (0.01) motivasyonu:** Neden bu değerler? Hiperparametre seçimi için deneysel gerekçe yok.
- **JointSelfAttention ablation yok:** "15 eklem küçük sequence → 1 katman yeterli" ifadesi kanıtsız. Self-attention'ın basit MLP decoder'a göre katkısı hiç ölçülmemiş. Attention mekanizması eklem arası bağımlılıkları öğreniyorsa bunun kanıtı nerede?
- **FiLM ablation yok:** FiLM conditioning obje geometrisini doğru biçimde enjekte etmek için seçilmiş; ancak basit concatenation veya addition yerine FiLM'in üstünlüğü sayısal olarak gösterilmemiş. Mimari kararın maliyeti var (parametre artışı, complexity) ama faydası kanıtsız.
- **Deterministik decoder vs. CVAE:** CVAE çıktısının (z sampled) basit bir deterministik decoder'a (z=0, mean pose) göre herhangi bir avantaj sağladığı gösterilmemiş. Unity export zaten z=0 kullanıyor — bu pratikte "CVAE kuruluyor ama deterministik çalışıyor" anlamına geliyor.
- **GRU seçimi:** Transformer/TCN varyantları neden değil? Theoretical motivation eksik.
- Centroid-proxy penetration loss (§10.9) tasarım hatası olduğu later sections'da açıkça kabul ediliyor; ancak neden başta bu kararın alındığı açıklanmamış.
- Contact loss hinge threshold (15mm) seçimi post-hoc gerekçelendirilmiş — başlangıç kararı olarak ikna edici değil.

---

## 4. Eğitim Protokolü — 6.5 / 10

**Güçlü yönler:**
- İki aşamalı eğitim (OakInk pretrain → HOT3D fine-tune) mantıklı ve uygulandı.
- Mixed DataLoader (%70 HOT3D / %30 OakInk replay) catastrophic forgetting'e karşı doğru önlem.
- β-KL warmup kullanımı standart pratikle uyumlu.
- Training log'ları eksiksiz ve dürüst (§4, §9).

**Zayıf yönler:**
- **Phase 2 yalnızca 14 epoch** — convergence için açıkça yetersiz (§6.1, §9.2'de de kabul ediliyor). Bu bir zaman/kaynak kısıtı ise tezde açıklanmalı.
- **Augmentasyon hiç uygulanmadı** (§11.8). Tanımlanmış ama implement edilmemiş 7 augmentasyon türü var. Augmentasyonsuz sonuçlar bir tez için zayıf zemin.
- **OakInk split sample-bazlı** (§11.2, §7.4): Aynı objenin farklı graspları train/test'e dağılabilir → genelleme testi anlamlı değil. Bu **metodolojik bir hata**, düzeltilmesi gerekiyor.
- **3 random seed** gerekliliği §5.6'da belirtilmiş ama uygulanmamış — tüm sonuçlar tek seed'den.

---

## 4b. Niteliksel Değerlendirme Eksikliği — 3 / 10

Tezde hiçbir görsel veya video çıktısı yok.

- **Resim:** Üretilen kavrama pozlarının 3D görselleştirmesi (MANO mesh + obje point cloud + predicted finger pose) yok. Contact ratio %13 bir sayı olarak sunuluyor; ama parmakların objenin neresine yaklaştığı, hangi grasp stratejisinin üretildiği görsel olmadan değerlendirilemiyor.
- **Video:** HOT3D temporal sekansları üzerinde model çıktısının frame-by-frame animasyonu yok. Jitter_score 6.15 sayısının ne anlama geldiği, gözle görülen titreme ne kadar ciddi — bilinmiyor.
- **Unity demo videosu:** Demo scene çalışıyor ama sistemi değerlendiren bir kayıt veya ekran görüntüsü sunulmamış.

XR el hareketi üzerine yapılan bir tezde niteliksel gösterim zorunludur. Sayısal metrik tablolarının yeterli olduğu varsayımı yanlış: bu alanda jüriler videoyu ister, çünkü "naturalistic" hareketin sayısal karşılığı tartışmalıdır.

---

## 5. Değerlendirme Protokolü — 7 / 10

**Güçlü yönler:**
- Geodesic rotation error, MPJPE, contact ratio, penetration, jitter metrikleri kapsamlı set oluşturuyor.
- Bootstrap CI ve paired bootstrap protokolü (§5.6) istatistiksel raporlama için doğru yaklaşım.
- 4-seviyeli HOT3D obje bölümü (train/val/calibration/held-out) doğru tasarım.
- Unity fizik eval protokolü detaylı tanımlanmış (§5.3).

**Zayıf yönler:**
- Geodesic 9.7° veya 12.6° değerlerinin iyi mi kötü mü olduğu **hiçbir yerde cevaplanmıyor**. SOTA karşılaştırması olmadan bu rakamlar anlamsız.
- Unity eval henüz çalıştırılmadı — sistemin asıl iddiası olan "XR'da çalışır grasp" doğrulanamadı.
- HOT3D val/test'te yalnızca 3–4 obje: bu kadar küçük örneklemle %95 CI hesaplamak yanıltıcı güven verir (§7.4'te kendisi de kabul ediyor).

---

## 6. Deneyler ve Sonuçlar — 4 / 10

Bu bölüm tezin en zayıf kısmıdır.

**Ana araştırma sorularının durumu:**

| Soru | Durum |
|------|-------|
| OakInk+HOT3D tek modelde birleşiyor mu? | Kısmen — pipeline çalışıyor |
| PointNet > BBox? | **Cevaplanmadı** |
| Temporal > SingleFrame? | **Cevaplanmadı** |
| CVAE K>1 başarıyı artırıyor mu? | Hayır (K=1/3/5 fark yok) |
| Confidence head başarısızları ayırt ediyor mu? | Kısmen hayır |
| Gerçek zamanlı çalışıyor mu? | **Cevaplanmadı** |

6 sorudan 1'i olumlu, 1'i olumsuz, 1'i kısmen — **4'ü hâlâ açık.**

**Kritik sayısal sorunlar:**

1. **Contact ratio: %13–23 vs. hedef %70** — Sistemin parmakları objeye temas ettirme kapasitesi çok yetersiz. Bu, fizik-based grasp'ın temel önkoşulunu karşılamıyor.
2. **Joint limit violation %38–54** — Üretilen pozların yarısı anatomik olarak imkânsız. Savunmada bu doğrudan sorulacak.
3. **CVAE diversity 0.051** — CVAE ilave değer üretmiyor; tek deterministic çıktıdan farklı değil.
4. **HOT3D quality_score Spearman 0.157** — Rastgele tahminden biraz iyi; kullanılabilir değil.
5. **success_prob head eğitilmedi** — Aday seçimi mekanizması işlevsiz.

**Yapılmamış ablation ve baseline deneyleri — tam liste:**

Mimarideki her bileşen için ya bir baseline ya da bir ablation gerekiyor. Hiçbiri yapılmadı:

| Bileşen | Gerekli karşılaştırma | Olmadan ne bilinemiyor? |
|---------|----------------------|------------------------|
| PointNet encoder | MLP-BBox baseline | Geometri bilgisinin katkısı |
| FiLM conditioning | FiLM → concat/add | FiLM'in concatenation'a göre faydası |
| GRU temporal encoder | SingleFrame (T=1) baseline | Temporal bağlamın katkısı |
| JointSelfAttention | Self-attn → basit MLP decoder | Eklem arası attention'ın katkısı |
| CVAE sampler | Deterministik decoder (z=0 sabit) | Stochastic sampling'in katkısı |
| K>1 aday seçimi | K=1 (mevcut sonuç aynı) | Multi-candidate selection'ın katkısı |
| success_prob head | (eğitilmedi) | Aday seçiminin herhangi bir faydası |
| Gerçek zamanlı inference | (ölçülmedi) | Sistemin XR hedefine uygunluğu |

Toplam 8 kritik karşılaştırmadan **0 tanesi tamamlandı.** Bu durum şunu gösteriyor: mimarinin her katmanı bağımsız olarak katkısı bilinmeyen bileşenlerden oluşuyor. Çalışan tek şey birleşik sistemin loss azaldığıdır — neyin neden azalttığı bilinmiyor.

**Goodhart's Law — Metrik Optimizasyonu vs. Gerçek Görev:**

Contact ratio %13 (düşük) ve penetration ~0.5mm (düşük) değerleri birlikte okunduğunda çelişik görünüyor; aslında tutarlı ve daha ciddi bir sorunu işaret ediyor: **model objeye hiç yaklaşmıyor.**

- Parmaklar objeye değmedi → contact ratio düşük ✓
- Parmaklar objenin içine girmedi → penetration düşük ✓
- Loss azaldı → model "başarılı" ✓
- Ama gerçek bir kavrama üretilmedi ✗

Bu klasik bir Goodhart's Law durumu: model L_contact ve L_penetration'ı minimize etmenin en kolay yolunu bulmuş — objeye yaklaşmamak. Hinge toleransı 15mm ve centroid-proxy SDF birlikte "uzakta dur, loss küçük olsun" kısayolunu mümkün kıldı. Contact ratio metriği aynı anda hem loss bileşeni hem de değerlendirme metriği olduğundan metrik optimize edildi ama görev çözülmedi.

Bu pattern §9.1 ve §12.1'de kısmen fark edilmiş ama "Goodhart tuzağı" olarak adlandırılmamış. Düzeltme yalnızca loss redesign değil, **loss ile metrik arasındaki döngüsel bağımlılığı kırmak** gerekiyor: bağımsız bir SDF-based contact metriği loss'tan ayrı tutulmalı.

**Olumlu bulgu:**
- Per-object analiz tablosu (§6.7) gerçek bir gözlem içeriyor ve failure pattern'leri anlamlı biçimde özetleniyor.

---

## 6b. Mimari Mantık Hatası — Approach-to-Grasp Geçişi

Bu tezde tartışılmayan ama XR kullanımının özünü etkileyen bir tasarım problemi var.

**Problem:** Model her timestep'te bir "final grasp pozu" üretiyor. Ama el objeye yaklaşırken parmakların doğal olarak açık olması gerekiyor; grasp pozu yalnızca temas anında ve sonrasında anlamlı.

**Ne oluyor runtime'da:**
1. El 8cm uzaktayken model zaten kapalı parmak pozu üretiyor (GRU temporal context'i grasp'a doğru iter).
2. Unity blend controller 10cm'de devreye giriyor ve bu kapalı pozu ağırlıklandırmaya başlıyor.
3. Kullanıcı, el henüz objeye değmemişken parmaklarının kapanmaya başladığını görüyor — doğal değil ve XR'da rahatsız edici.

**Neden oluyor:**
- Model mimarisi approach fazı ile grasp fazını birbirinden **hiç ayırt etmiyor.** Tek bir decoder her zaman 45-dim parmak pozu üretiyor.
- HOT3D eğitim verisinin ~%79'u quality_label=0 (approach fazı) — GT parmak pozu bu framelerde açık el. Ama 14 epoch fine-tuning ile model bu faz ayrımını öğrenememiş olabilir; loss'ta approach ve grasp eşit ağırlıkla görünüyor.
- ONNX export z=0 (deterministik mean pose) kullanıyor. Bu mean pose, approach (~%79) ve grasp (~%21) framelerinin ağırlıklı ortalaması; ne tam açık el ne tam kapalı — her zaman yarı kapalı bir şey üretiyor.
- contact_flag sinyali bu faz ayrımını öğretmek için var ama contact_ratio %13 ile bu sinyalin öğrenildiğinden şüphe duyulabilir.

**Yapısal çözüm (tartışılmadı):**
- Model çıktısına **faz koşullaması** eklenmeli: `dist` veya `contact_flag`'e göre decoder'ın "approach modu" (açık el) ile "grasp modu" (kapalı el) arasında geçiş yapması.
- Alternatif: approach fazı için ayrı bir basit model (el açık tutma) + yalnızca temas yakınında grasp decoder devreye girme.
- Unity blending bunu kısmen maskeliyor ama model düzeyinde çözülmeden gerçek doğallık sağlanamaz.

**Tezde nerede eksik:** §3.1 "kural tabanlı faz geçişi + AI grasp decoder ayrımı" olarak sunulmuş — bu ayrım iyi bir tasarım gibi gösteriyor. Ama ayrımın sınırı yanlış çizilmiş: AI decoder approach fazı sırasında da çalışıyor ve yanlış çıktı üretiyor. §7.5 sistem tasarım kararlarında bu geçiş davranışı "Unity demo'da henüz test edilmedi" olarak geçiştiriliyor; oysa bu mimari düzeyde bir sorundur.

---

## 7. Tartışma ve Sonuç — 7 / 10

**Güçlü yönler:**
- §7 son derece dürüst: kendi başarısızlıklarını açıkça adlandırıyor ve "neden çalışmadı" sorusuna teknik yanıt vermeye çalışıyor.
- §8.3'teki "Kritik açık problemler" listesi doğru önceliklendirme yapıyor.
- Contact loss redesign ve joint limit güçlendirme önerileri somut ve uygulanabilir.

**Zayıf yönler:**
- Sonuç bölümü (§8) "bu sistem çalışıyor" demek yerine "bu sistem büyük ölçüde çalışmıyor ve işte nedenler" diyor — bu **dürüstlük değerli** ama bir tez için nihai katkıyı zayıflatıyor.
- Gelecek çalışmalar listesi mevcut tezin tamamlanmamış parçaları gibi görünüyor (PointNet++, kullanıcı çalışması, Quest deployment) — bunlar tezin kapsamının dışında mı, yoksa yarım kalan kısımlar mı, netlik yok.

---

## 8. Teknik Uygulama Kalitesi — 7.5 / 10

**Güçlü yönler:**
- §12'deki bug fix kaydı çok değerli: FK hatası (18cm → 0.9cm), contact threshold uyumsuzluğu, temporal loss sessiz devre dışı kalma — bunlar production-level mühendislik deneyimi gösteriyor.
- ONNX export ve Unity InferenceEngine GRU unroll çözümü özgün problem çözme.
- İki farklı ONNX versiyonu (araştırma vs. Unity) tasarımı doğru.
- Centroid-proxy penetration'ın neden artifact ürettiğini anlayıp §12.9'da kaydetmiş olması farkındalık gösteriyor.

**Zayıf yönler:**
- FK bug'ının eğitimin bu kadar geç aşamasında tespit edilmesi, validation pipeline'ının yetersizliğini gösteriyor.
- contact_flag'in yanlış eşikle kullanılması (5mm vs 30mm) veri pipeline doğrulamasının eksikliğini ortaya koyuyor.
- Phase 2 temporal loss'un "sessizce devre dışı" kaldığı bug (§12.6) ciddi — bu tür hatalar erken yakalanmalı.

---

## 9. En Kritik Eksiklikler (Savunmada Kesinlikle Sorulacaklar)

1. **"BBox baseline olmadan PointNet'in katkısını nasıl kanıtlıyorsunuz?"**  
   Cevap: Kanıtlanamıyor. En önemli ablation deneyi yapılmadı.

2. **"SingleFrame baseline olmadan temporal modelin katkısını nasıl gösteriyorsunuz?"**  
   Cevap: Gösterilemiyor. Jitter sayısının ne ifade ettiği bilinmiyor.

3. **"Contact ratio %13 ile bu sistem gerçek bir XR uygulamasında nasıl kullanılacak?"**  
   Cevap: Mevcut haliyle kullanılamaz. Kritik tasarım hatası (centroid-proxy) kabul edilmiş ama düzeltilmemiş.

4. **"OakInk split'inizde veri sızıntısı var mı?"**  
   Cevap: Sample-bazlı split nedeniyle aynı objenin graspları farklı setlere dağılmış. Gerçek genelleme testi yapılmamış.

5. **"Unity fizik değerlendirmesi olmadan gerçek zamanlı XR iddiasını nasıl destekliyorsunuz?"**  
   Cevap: Desteklenemiyor. Tezin ana hedefi (XR'da çalışan grasp) doğrulanmamış.

6. **"CVAE kullanmanızın faydası nedir? K=1 ile K=5 arasında hiç fark yok."**  
   Cevap: Diversity 0.051 ile CVAE'nin katkısı kanıtlanamamış; KL weight çok düşük. Unity export zaten z=0 kullanıyor, yani production sistemde deterministik decoder ile özdeş.

7. **"JointSelfAttention'ın katkısı nedir? Basit bir MLP decoder'dan neden üstün?"**  
   Cevap: Ablation yok. Self-attention yerine per-joint bağımsız MLP ile karşılaştırma yapılmamış. Eklem arası bağımlılığın öğrenilip öğrenilmediği bilinmiyor.

8. **"FiLM'i seçmenizin gerekçesi nedir? Concatenation da çalışmaz mıydı?"**  
   Cevap: FiLM vs. concat/add ablation yok. Ek parametre ve karmaşıklığın somut faydası gösterilmemiş.

9. **"Sisteminiz gerçek zamanlı çalışıyor mu? Latency hedefi neydi?"**  
   Cevap: Latency hiç ölçülmedi. Hedef <5ms (§5.5) ama mevcut inference süresi bilinmiyor. Tezin "gerçek zamanlı XR" iddiası ölçümsüz kalıyor.

10. **"success_prob head'i olmadan aday seçimi nasıl yapılıyor?"**  
    Cevap: Eğitilmemiş bir head rastgele sinyale yakın çıktı veriyor. K=1/3/5 karşılaştırmasındaki "fark yok" sonucunun nedeni zaten budur — seçim mekanizması işlev dışı.

11. **"El objeye yaklaşırken neden parmaklar kapanmaya başlıyor? Bu kasıtlı bir tasarım mı?"**  
    Cevap: Hayır, kasıtlı değil — mimari bir mantık hatasının sonucu. Model her timestep'te final grasp pozu üretiyor; approach fazı ile grasp fazını ayırt etmiyor. Blend controller 10cm'de devreye girdiğinde model zaten kapalı el üretiyor, kullanıcı objeye değmeden parmaklarının kapandığını görüyor. (Bkz. §6b)

12. **"Neden hiç görsel veya video yok? Sisteminizin ürettiği kavramalar neye benziyor?"**  
    Cevap: Tezde niteliksel gösterim bulunmuyor. Contact ratio %13'ün ne anlama geldiği, jitter'ın gözle görülüp görülmediği, parmakların objeye nasıl yerleştiği — bunların hiçbiri gösterilmemiş.

---

## 10. Özet Puanlama

| Kriter | Puan (10 üzerinden) |
|--------|---------------------|
| Problem tanımı ve motivasyon | 7.0 |
| İlgili çalışmalar | 5.0 |
| Yöntem tasarımı | 7.5 |
| Eğitim protokolü | 6.5 |
| Değerlendirme tasarımı | 7.0 |
| Deneysel bulgular | 4.0 |
| Tartışma dürüstlüğü | 8.0 |
| Teknik uygulama | 7.5 |
| **Genel Ortalama** | **6.6** |
| **Bilimsel katkı kanıtı** | **4.0** |
| **Genel Tez Puanı** | **5.5** |

> Bilimsel katkı kanıtı ayrı tutulmuştur çünkü teknik uygulama kalitesi ile bilimsel doğrulama tamamlanmışlığı arasında belirgin açıklık vardır.

---

## 11. Tezi Kabul Ettirecek Minimum Gereksinimler

Aşağıdakiler tamamlanmadan tez savunmaya alınamaz:

- [ ] **SingleFrame baseline** deneyi — temporal katkının kanıtı için zorunlu
- [ ] **MLP-BBox baseline** deneyi — PointNet katkısının kanıtı için zorunlu
- [ ] **OakInk object-level split** düzeltmesi — metodolojik hata
- [ ] **Contact loss redesign** (gerçek mesh SDF) + yeniden eğitim — %13 contact ratio kabul edilemez
- [ ] **Joint limit violation** düşürülmesi (%54 → hedef <%10)
- [ ] **En az 3 random seed** ile tüm main varyantların yeniden çalıştırılması
- [ ] **Unity fizik eval** veya en azından offline metrik SOTA karşılaştırması
- [ ] **SingleFrame baseline** — temporal katkısı ölçülmeden temporal model iddiası kanıtsız
- [ ] **MLP-BBox baseline** — geometri katkısı ölçülmeden PointNet seçimi kanıtsız
- [ ] **Deterministik decoder (z=0 sabit) baseline** — Unity export zaten bunu kullanıyor; CVAE'nin katkısı sıfır mı gerçekten?
- [ ] **JointSelfAttention ablation** (self-attn → basit MLP) — attention'ın katkısını göster
- [ ] **Latency benchmark** (PC GPU, warm-up sonrası 100 iter) — "gerçek zamanlı" iddiası için zorunlu
- [ ] **Approach-to-grasp faz ayrımı** — model approach fazında yanlış pose üretiyor; mimaride ya faz koşullaması ya da ayrı approach modu gerekiyor
- [ ] **Niteliksel gösterim** — en az 5–10 obje için predicted grasp render'ı ve bir Unity demo videosu

---

## 12. Tezi Güçlendirecek Öneriler (Minimum Üstü)

- **CVAE KL weight artışı** (0.01 → 0.1) + diversity analizi
- **Phase 2 uzun eğitim** (14 → 50+ epoch)
- **Augmentasyon uygulama** (en az yaw augmentation ve velocity perturbation)
- **HOT3D thumb DOF** etkisinin quantify edilmesi
- **Latency benchmark** (PC GPU, 100 iter ortalaması)
- **SOTA karşılaştırma tablosu** (GrabNet veya ContactOpt ile OakInk geodesic karşılaştırması)

---

## Sonuç

Bu tez, gerçek bir mühendislik problemi üzerine iyi düşünülmüş bir mimari geliştirmiş ve süreci son derece dürüst bir biçimde kayıt altına almıştır. Özellikle §12'deki bug fix kaydı ve §7'deki başarısızlık analizi, araştırmacının teknik olgunluk kazandığını gösteriyor. Ancak akademik standartta bir master tezi için gerekli olan **deneysel kanıt** büyük ölçüde eksiktir: en önemli ablation deneyleri yapılmamış, ana metrik hedefi karşılanmamış, sistemin asıl kullanım senaryosundaki (XR fizik eval) performansı ölçülmemiştir.

**Mevcut haliyle tez "çok iyi bir araştırma notu" kalitesindedir; ancak savunmaya hazır bir akademik tez değildir.**
