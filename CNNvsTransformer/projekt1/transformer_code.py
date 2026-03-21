#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
#from einops import rearrange, repeat
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
device


# Poniższa klasa dzieli nasze obrazy na tzw. patche z embeddingami i dodaje token CLS. Dodaje również informację o pozycji każdego patcha.
# 
# W konstruktorze image_size to rozmiar obrazków (256x256), patch_size to rozmiar patcha (na razie 16 - to w sumie strzał, może być do zmiany), channels to liczba kanałów (dla RGB to 3), a embedding_size to wymiar wektora (embeddingu) dla każdego patcha. Generalnie rodzicem tej klasy jest nn.Module. 
# 
# Po kolei w funkcji forward:
# 1. Wchodzi jakiś x=[ilość obrazów, ilość kanałów, wysokość, szerokość] (u nas [b,3,256,256])
# 2. Robimy konwolucję (self.projection) - przesuwamy jądro 16x16 po obrazie co 16 pikseli, dostajemy taką siatkę "kafelków" 16x16 i każda jest opisana przez 256 liczb (bo embedding_size=256) (dostajemy [b,256,16,16])
# 3. Zmieniamy trochę kształt - zlepiamy wysokość i szerokość w jeden wymiar (h w)=16x16=256 i mamy jeden wiersz z liczbami (dostajemy [b,256,256])
# 4. Powielamy token CLS dla każdego obrazu i doklejamy wzdłuż dim=1 (to drugi wymiar) (czyli [patch1,patch2,...]->[CLS,patch1,patch2,...]) (dostajemy [b,1,256]+[b,256,256]=[b,257,256])
# 5. Dodajemy informację o pozycji patchy (na początku są losowe, ale model się ich potem uczy) (dostajemy [b,257,256])
# 6. Zwracamy x

# In[10]:


class PatchEmbedding(nn.Module):

    def __init__(self, image_size=256, patch_size=16, channels=3, embedding_size=256):

        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = channels
        self.embedding_size = embedding_size

        self.n_patches = (image_size//patch_size) ** 2

        # dzielimy obraz na 256 patchy 16x16
        self.projection = nn.Conv2d(
            in_channels = channels,
            out_channels = embedding_size,
            kernel_size = patch_size, # jadro ma pokryc caly patch - rozmiar okienka (patch_size x patch_size)
            stride = patch_size # nie chcemy zeby patche na siebie nachodzily, nasz "skok" - o ile sie przesunac w bok i w dol
         )

        # token CLS
        self.CLS_token = nn.Parameter(torch.randn(1, 1, embedding_size))

        # embedding pozycyjny
        self.positions = nn.Parameter(torch.randn(self.n_patches + 1, embedding_size))

    def forward(self, x):

        batch_size = x.shape[0]

        x = self.projection(x) # tu zamieniamy (batch_size, channels, image_size, image_size) na (batch_size, embedding_size, patch_size, patch_size)

        # to jest troche magia, ale zmienia ksztalt na taki jaki chcemy
        # x = rearrange(x, 'b e h w -> b (h w) e') - czli zostawiamy batch_size na poczatku, wymiary mnozymy i dajemy do srodka, wymiar embeddingu na koniec 
        x = x.flatten(2).transpose(1, 2) # flatten robi nam jeden rozmiar (hw), transpose przenosi nam wymiar embeddingu na koniec

        #CLS_tokens = repeat(self.CLS_token, '1 1 d -> b 1 d', b=batch_size)
        CLS_tokens = self.CLS_token.expand(batch_size, -1, -1) 
        x = torch.cat([CLS_tokens, x], dim=1)

        x = x + self.positions

        # ogolnie infomacje o tym co jest na obrazku - daje nam projection, a gdzie kafelek jest - daje nam positions

        return x


# In[11]:


# test wymiarow

embed = PatchEmbedding(image_size=256, patch_size=16, channels=3, embedding_size=256)
x = torch.randn(4, 3, 256, 256)
output = embed(x)
print(f"output: {output.shape}")


# Dostajemy x - jakiś tensor po patch embeddingu. Jest postaci [b,257,256]. Mamy 257 tokenów (bo 1 CLS i 256 patchy), a każdy token to wektor 256 liczb. Po kolei:
# 1. Robimy query, key i value. Wszystkie mają kształt [b,257,64].
# 2. Transponujemy key (żeby dało się mnożyć) i mnożymy query i transponowane key. Dostajemy [b,257,257]. Generalnie wartość [i,j] dla to iloczyn skalarny query[i] i key[j]. Skalujemy - zeby liczby byly mniejsze i stabilniejsze (duze liczby moglyby popsuc dzialanie softmaxu).
# 3. Robimy softmax. dim=-1 gwarantuje, że dla każdego wiersza suma musi być 1.
# 4. Dropout - zapobiega overfittingowi.
# 5. Mnożymy przez value. To jest informacja jak bardzo token ma patrzeć na inne tokeny (jak dużo attention im dać).
# 6. Zwracamy dwie rzeczy - attention_out ([b,257,64], nowe reprezentacje tokenów, które uwzględniają kontekst) i attention_probs ([b,257,257], na jakie patche inne patche zwracają uwagę).

# In[12]:


class AttentionHead(nn.Module):

    def __init__(self, embedding_size=256, attention_head_size=64, dropout=0.2):

        super().__init__()

        self.embedding_size = embedding_size
        self.attention_head_size = attention_head_size

        # ponizej tworzymy trzy warstwy (Q,K,V) za pomoca zwyklych warstw liniowych

        self.query = nn.Linear(embedding_size, attention_head_size) # ile wchodzi - embedding_size, ile ma wyjsc - attention_head_size (zmiana dotyczy kolumn (czyli "cechy - liczby w wektorze"))
        self.key = nn.Linear(embedding_size, attention_head_size)
        self.value = nn.Linear(embedding_size, attention_head_size)

        # dzieki tym powyzszym warstwom to tak jakby bierzemy te 256 surowych liczb "cech", i otrzymujemy 64 nowe, "madrzejsze" cechy

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.attention_head_size) # transpose(-1,-2) zamienia miejscami ostatnie dwa wymiary tensora
        # powyzej - te mnozenie to tutaj ustalam relacje miedzy patchami - wagi
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_out = attention_probs @ value # ile brac z wektora value - zalezne od wag 

        return attention_out, attention_probs


# In[13]:


# test wymiarow

head = AttentionHead(embedding_size=256, attention_head_size=64)
x = torch.randn(4, 257, 256)
out, probs = head(x)

print(f"attention_out: {out.shape}")
print(f"attention_probs: {probs.shape}")


# Poniższa klasa łączy kilka głów attention (w tym przypadku 4 - może do zmiany?). Wszystkie po kolei analizują te same dane (ale mogą "zwracać uwagę" na różne aspekty tych danych). Łączymy wyniki wzdłuż ostatniego wymiaru i przepuszczamy przez warstwe liniową i warstwę z dropoutem.

# In[14]:


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_size=256, n_heads=4, dropout=0.2):

        super().__init__()

        self.n_heads = n_heads
        self.embedding_size = embedding_size
        self.head_size = embedding_size//n_heads

        self.heads = nn.ModuleList() # zeby pytorch widzial te glowy - to taka specjalna lista (glowic)
        for _ in range(n_heads):
            self.heads.append(
                AttentionHead(embedding_size=embedding_size, attention_head_size=self.head_size, dropout=dropout)
            )

        # kazda glowica zaczyna z innym zestawem liczb - innymi wagami

        self.combined = nn.Linear(embedding_size, embedding_size) # warstwa liniowa - sluzy do wymiany danych miedzy glowicami
        self.dropout=nn.Dropout(dropout)


    def forward(self, x, output_attentions=False):

        all_attentions = []
        heads_out = [] # outputy z glow

        for head in self.heads:
            if output_attentions:
                out, attention_probs = head(x, output_attentions=True)
                heads_out.append(out)
                all_attentions.append(attention_probs)
            else:
                out, _ = head(x) # wagi ignoruje
                heads_out.append(out)

        together = torch.cat(heads_out, dim=-1) # 4 glowice sklejam ze soba 

        out = self.combined(together) # wymiana informacji ktore pochodza z 4 roznych glowic - ale juz wewnatrz jednego tensora (po sklejeniu tych 4 glowic wymiana info)
        out = self.dropout(out)

        if output_attentions:
            return out, all_attentions

        return out


# In[15]:


mha = MultiHeadAttention(embedding_size=256, n_heads=4, dropout=0.2)
x = torch.randn(4, 257, 256)
out = mha(x)

print(f"wejscie: {x.shape}")
print(f"wyjscie: {out.shape}")


# Dostajemy x - jakiś tensor po multiheadattention. Jest postaci [b,257,256]. Mamy 257 tokenów (bo 1 CLS i 256 patchy), a każdy token to wektor 256 liczb. MLP (MultiHeadPerceptron) generalnie zwiększa pojemność modelu i umożliwia mu uczenie się bardziej skomplikowanych wzorców. Po kolei:
# 1. Powiekszamy "przestrzeń" cech (w sensie wymiar (liczby w wektorze) kazdego tokena) - dzięki temu model może tworzyć bardziej złozone kombinacje cech.
# 2. Nakładamy funkcje aktywacji i wprowadzamy nieliniowość.
# 3. Wracamy do pierwotnego wymiaru cech.
# 4. Dropout - zapobiega overfittingowi.

# In[16]:


class MLP(nn.Module):

    def __init__(self, embedding_size=256, dropout=0.2):

        super().__init__()

        intermediate_size = 4 * embedding_size # tak dziala po prostu standardowy transformer - potrzebuje wiecej przestrzeni, dzieki rozszerzeniu model moze sie uczyc bardziej zlozonych zaleznosci

        self.dense_1 = nn.Linear(embedding_size, intermediate_size) # pierwsza warstwa - zwiększamy wymiar ( u nas *4, czyli z 256 na 1024)

        self.activation = nn.GELU() # GELU to taka lepsza wersja ReLU, GELU stopniowo wygasza ujemne wartosci, a nie odcinka od razu jak w przypadku ReLU

        self.dense_2 = nn.Linear(intermediate_size, embedding_size) # druga warstwa - wracamy do oryginalnego rozmiaru (256)

        self.dropout = nn.Dropout(dropout) # zapobiegamy przeuczeniu

    def forward(self, x):

        x = self.dense_1(x) # rozszerzamy przestrzen 
        x = self.activation(x)

        x = self.dense_2(x) # powrot do pierwotnego rozmiaru
        x = self.dropout(x)

        return x


# In[17]:


mlp = MLP(embedding_size=256, dropout=0.2)
x = torch.randn(4, 257, 256)
out = mlp(x)

print(f"MLP wejscie: {x.shape}") 
print(f"MLP wyjscie: {out.shape}") 


# Klasa TransformerBlock łączy wszystko co do tej pory napisaliśmy.
# 1. Normalizacja.
# 2. MultiheadAttention.
# 3. Normalizacja.
# 4. MLP.

# In[18]:


class TransformerBlock(nn.Module):

    def __init__(self, embedding_size=256, n_heads=4, dropout=0.2):

        super().__init__()

        self.attention = MultiHeadAttention(embedding_size=embedding_size, n_heads=n_heads)

        self.layernorm_1 = nn.LayerNorm(embedding_size) # normalizacja przed attention

        self.mlp = MLP(embedding_size=embedding_size, dropout=dropout)

        self.layernorm_2 = nn.LayerNorm(embedding_size) # normalizacja przed MLP

    def forward(self, x, output_attentions = False):

        if output_attentions:
            attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=True)
        else:
            attention_output = self.attention(self.layernorm_1(x)) # najpierw normalizujemy x, potem liczymy attention# najpierw normalizujemy x, potem liczymy attention

        # tu teraz ponizej to skip connection
        x = x + attention_output # czyli wynik - to oryginalne wejscie + wynik z attention
        # dodajemy oryginalne wejscie do wyniku uwagi
        # to generalnie pomaga nam zapobiegac problemowi zanikajacego gradientu
        # (jak tego nie ma to mnozymy wiele razy przez male liczby, wiec gradient robi sie malutki)

        mlp_output = self.mlp(self.layernorm_2(x)) # najpierw normalizujemy x, potem MLP

        x = x + mlp_output # skip connection (i po attention)

        if output_attentions:
            return x, attention_probs
        return x


# In[19]:


block = TransformerBlock(embedding_size=256, n_heads=4, dropout=0.2)
x = torch.randn(4, 257, 256)
out = block(x)

print(f"TransformerBlock wejscie: {x.shape}")  
print(f"TransformerBlock wyjscie: {out.shape}")  


# Encoder to w zasadzie taki stos bloków transformera. Ważne - każdy blok ma swoje własne wagi!

# In[20]:


class Encoder(nn.Module):

    def __init__(self, embedding_size=256, num_layers=4, n_heads=4, dropout=0.2):
        # num_layers - my decydujemy o ilosci warstw takze mozliwe ze bedzie do zmiany 
        super().__init__()

        # nn.ModuleList - to znow ta taka "lepsza lista", lista warstw - PyTorch dzieki temu widzi te bloki (warstwy) i moze trenowac ich wagi
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_size=embedding_size, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers) # czyli mamy nasz zestaw operacji (A+MLP) i num_layers mowi ile razy ten zestaw operacji wykonac na danych (za kazdym razem uzyjemy innych wag - to bedzie ywkrywac rozne rzeczy)
        ])

    def forward(self, x, output_attentions=False): # true sie nam przyda dopiero przy analizie 
        all_attentions = [] # dla kazdej warstwy bedziemy mieli zapisane probs 

        for block in self.blocks:
            if output_attentions:
                x, attention_probs = block(x, output_attentions=True)
                all_attentions.append(attention_probs)
            else:
                x = block(x, output_attentions=False) # bez wyciagania probs

        if output_attentions:
            return x, all_attentions # to sie przyda potem do analizy 
        else:
            return x


# In[21]:


encoder = Encoder(embedding_size=256, num_layers=4, n_heads=4, dropout=0.2)
x = torch.randn(4, 257, 256)
out = encoder(x)

print(f"Encoder wejscie: {x.shape}") 
print(f"Encoder wyjscie: {out.shape}")  


# Po kolei:
# 1. PatchEmbedding - dzielimy obrazek na patche i zamieniamy je na wektory + informacja o pozycji patchy + doklejamy token CLS na początek. 
# 2. Encoder - przechodzimy zestaw operacji (A+MLP) tyle razy, ile mamy warstw (bloków).
# 3. Wycinamy token CLS - tylko on nam jest potrzebny.
# 4. Klasyfikator - zamieniamy wketor 256 liczb na 2 wyniki - logity.

# In[22]:


class ViT(nn.Module):

    def __init__(self, image_size=256, patch_size=16, channels=3, embedding_size=256, num_layers=4, n_heads=4, num_classes=2, dropout=0.2):

        super().__init__()

        # tworzenie embeddingow - pociecie obrazka i dodanie tokena CLS
        self.embedding = PatchEmbedding(
            image_size=image_size, 
            patch_size=patch_size, 
            channels=channels,
            embedding_size=embedding_size
        )

        # encoder
        self.encoder = Encoder(
            embedding_size=embedding_size, 
            num_layers=num_layers, 
            n_heads=n_heads, 
            dropout=dropout
        )

        # klasyfikator - interesuje go tylko wynik dla tokena CLS
        # zamienia embedding_size (256) na liczbę klas (2 - normal albo pneumonia) - czyli tlumaczy 256 liczb ktore dostajemy w wektorze na 2 klasy zeby bylo to dla nas zrozumiale 
        # te dwie liczby ktore dostaniemy na wyjsciu to tzw. logity (czyli jeszcze nie prawdopodobienstwa)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, output_attentions=False):
        # x to obrazek

        x = self.embedding(x) # zamiana obrazka na kafelki i token CLS

        if output_attentions:
            x, all_attentions = self.encoder(x, output_attentions=True)
        else:
            x = self.encoder(x, output_attentions=False) # przejscie przez encoder
            all_attentions = None

        # x ma wymiar [b, 257, 256], : - oznacza ze bierzemy wszystkie obrazki z batcha i drigi raz : - bierzemy wszystko z embedding_size
        cls_token_final = x[:, 0, :] # interesuje nas tylko token CLS, bierzemy 0 - bo token CLS jest na poczatku 

        logits = self.classifier(cls_token_final) # zamiana na 2 klasy (dostajemy 2 wyniki - logity (w sensie liczby, a nie nazwy klas typu normal/pneumonia)), wyjscie - wektor [batch_size,2]

        if output_attentions:
            return logits, all_attentions
        else:
            return logits


# In[23]:


model = ViT(image_size=256, patch_size=16, channels=3, embedding_size=256, num_layers=4, n_heads=4, num_classes=2)

images = torch.randn(4, 3, 256, 256)
logits = model(images)

print(f"ViT wejście (obrazy): {images.shape}") 
print(f"ViT wyjście (logity): {logits.shape}")  


# In[ ]:




