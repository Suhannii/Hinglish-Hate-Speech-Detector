"""
dataset_generator.py
--------------------
Generates a Hinglish hate speech dataset that includes:
- Community/group targeted hate
- Real Hindi abusive slang (gaali) in context
- Mixed Hinglish offensive sentences
- Clean non-hate sentences
"""

import pandas as pd
import os
import random

random.seed(42)

# ── Hate samples — includes real Hindi abusive slang in context ───────────────
HATE_SAMPLES = [
    # Group-targeted hate
    "Yeh log bahut gande hain, inhe yahan se nikalo",
    "Tum log kisi kaam ke nahi ho, chale jao",
    "Is community ke log sab chor hote hain",
    "Inhe is desh mein rehne ka haq nahi",
    "Yeh sab bakwaas log hain, koi value nahi inki",
    "Tum jaisi auratein ghar mein rehni chahiye",
    "Yeh religion wale sab terrorist hote hain",
    "Inhe maar do, desh ke liye achha hoga",
    "Yeh log hamare desh ko barbaad kar rahe hain",
    "Is caste ke log hamesha neeche hi rahenge",
    "Yeh sab illegal hain, inhe bahar karo",
    "Inhe vote dene ka haq nahi hona chahiye",
    "Yeh sab desh ke dushman hain",
    "Yeh community wale sab ek jaise hote hain, kharab",
    "Inhe school mein nahi aane dena chahiye",
    "Yeh religion ke log desh ke liye khatarnak hain",
    "Inhe yahan se bhaga do, hamare area mein nahi chahiye",
    "Yeh sab milke hamare desh ko todna chahte hain",
    "Is state ke log sab chor aur beimaan hote hain",
    "Yeh community ne desh ko barbaad kiya hai",
    "Inhe koi naukri nahi milni chahiye",
    "Yeh log hamare culture ko kharab kar rahe hain",
    "Inhe is sheher mein rehne nahi dena chahiye",
    "Yeh sab ek conspiracy ka hissa hain",
    "Tum log hamesha society ke liye burden rahoge",
    "Inhe bahar nikalo, hamare desh mein jagah nahi",
    "Yeh log hamare bacchon ke liye khatarnak hain",
    "Yeh sab log hamare tax ka paisa waste karte hain",
    "Tum log sirf desh ko nuksaan pahunchate ho",
    "Yeh log hamare desh ko divide karna chahte hain",

    # Direct abusive slang — single words / short phrases
    "madarchod",
    "bhenchod",
    "chutiya",
    "harami",
    "kamina",
    "kutte",
    "suar",
    "randi",
    "gaandu",
    "bhadwa",
    "ullu ka pattha",
    "saala kutta",
    "haramzada",
    "nalayak",
    "bevakoof kahin ka",

    # Abusive slang in sentence context
    "Tu bilkul madarchod hai, kuch nahi aata tujhe",
    "Yeh sab bhenchod log hamare desh ko barbaad kar rahe hain",
    "Teri maa ki aankh, chup kar",
    "Saale chutiye, kuch kaam kar",
    "Yeh harami log hamesha jhooth bolte hain",
    "Kamina insaan hai yeh, kisi pe bharosa mat karo",
    "Kutte ki tarah bhag gaya woh",
    "Haramzade ko koi kaam nahi dena chahiye",
    "Nalayak kahin ka, kuch nahi kar sakta",
    "Gaandu hai yeh banda, mat suno isko",
    "Bhadwa hai saala, kisi pe trust mat karo",
    "Ullu ka pattha hai, kuch samajh nahi aata isko",
    "Randi ki aulaad, desh chhod ke ja",
    "Suar ki tarah rehta hai yeh",
    "Yeh sab chutiye hain, inki baat mat suno",
    "Madarchod community ke log sab ek jaise hote hain",
    "Bhenchod yeh log hamare desh ke liye khatarnak hain",
    "Harami log hamesha aise hi karte hain",
    "Kamino ko is desh mein rehne ka haq nahi",
    "Saale kutte, apni aukat mein reh",
    "Yeh haramzade log hamesha desh ko nuksaan pahunchate hain",
    "Chutiya samajh rakha hai kya mujhe",
    "Teri maa ko gaali deta hoon main",
    "Bhosdike, kuch kaam kar apna",
    "Mc bc yeh log bahut gande hain",
    "Yeh sala kutta hai, kisi kaam ka nahi",
    "Gaali dene layak bhi nahi hai yeh",
    "Randi jaise kaam karta hai yeh",
    "Suar ki aulad, nikal yahan se",
    "Yeh sab madarchod hain, inhe nikalo",

    # Mixed English-Hindi abusive
    "You are a complete chutiya, get out",
    "Saale harami, you don't deserve to live here",
    "These bhenchod people are ruining everything",
    "What a kamina person, no one should trust him",
    "Yeh sab kutte hain, they should be thrown out",
    "He is a total gaandu, don't listen to him",
    "Haramzada sala, he deserves nothing",
    "These nalayak people are a burden on society",
    "Madarchod log, get out of our country",
    "Yeh sab chutiye hain, they are useless",
    "Such a bhadwa community, always causing trouble",
    "Ullu ke patthe, they understand nothing",
    "These suar log are destroying our culture",
    "Kamino ki aulad, they should be banned",
    "Randi ki tarah kaam karte hain yeh log",

    # Threatening / violent hate
    "Inhe maar dalo, desh ke liye zaroori hai",
    "Yeh log zinda rehne layak nahi hain",
    "Inhe khatam kar do, tab desh theek hoga",
    "Yeh sab ko jail mein daal do",
    "Inhe desh se nikal do ya maar do",
    "Yeh log deserve karte hain saza",
    "Inhe sabak sikhao, bahut ho gaya",
    "Yeh sab ko khatam karna chahiye",
    "Inhe barbad kar do, yahi sahi hai",
    "Yeh log deserve karte hain maut",
]

# ── Non-hate / neutral samples ────────────────────────────────────────────────
NON_HATE_SAMPLES = [
    "Aaj mausam bahut achha hai, bahar jaana chahiye",
    "Yeh movie bahut achi thi, tumhe dekhni chahiye",
    "Kal school mein exam hai, padhai karni hai",
    "Mujhe chai bahut pasand hai, especially masala chai",
    "Yeh restaurant ka khana bahut tasty hai",
    "Aaj cricket match dekhne ka plan hai",
    "Mera dost bahut helpful hai, hamesha saath deta hai",
    "Is project mein bahut kuch seekhne ko mila",
    "Yeh book bahut informative hai, padho zaroor",
    "Aaj gym gaya tha, bahut achha feel ho raha hai",
    "Yeh festival bahut colorful hota hai, mujhe pasand hai",
    "Mere ghar mein sab theek hain, shukriya poochne ke liye",
    "Yeh technology bahut useful hai daily life mein",
    "Aaj office mein presentation achi gayi",
    "Mujhe travel karna bahut pasand hai",
    "Yeh song bahut achha hai, baar baar sunta hoon",
    "Aaj market se sabzi lani hai",
    "Mere teacher bahut achhe hain, clearly samjhate hain",
    "Yeh app bahut useful hai, try karo",
    "Aaj dost ke saath movie dekhne ja rahe hain",
    "Mujhe coding bahut interesting lagti hai",
    "Yeh jagah bahut sundar hai, photos leni chahiye",
    "Aaj khana ghar par bana, bahut tasty tha",
    "Mere bhai ne exam mein achhe marks laye",
    "Yeh problem solve ho gayi, bahut achha laga",
    "Aaj yoga kiya, bahut relaxed feel ho raha hai",
    "Mujhe is subject mein interest hai",
    "Yeh design bahut creative hai",
    "Aaj rain ho rahi hai, ghar mein rehna better hai",
    "Mere dost ki birthday hai aaj, party hai",
    "Yeh game bahut fun hai, sab ko khelna chahiye",
    "Aaj library mein gaya, bahut books hain",
    "Mujhe painting karna pasand hai",
    "Yeh news bahut important hai, padho",
    "Aaj college mein cultural event tha, bahut maza aaya",
    "Mere parents bahut supportive hain",
    "Yeh recipe try karni chahiye, bahut easy hai",
    "Aaj morning walk pe gaya, fresh feel ho raha hai",
    "Mujhe music sunna bahut pasand hai",
    "Yeh documentary bahut informative thi",
    "Aaj dost se milne ka plan hai",
    "Mere ghar ke paas ek achha park hai",
    "Yeh course bahut helpful hai career ke liye",
    "Aaj shopping karne gayi, bahut sari cheezein mili",
    "Mujhe gardening bahut pasand hai",
    "Yeh event bahut well-organized tha",
    "Aaj office mein team lunch tha, bahut maza aaya",
    "Mere neighbour bahut friendly hain",
    "Yeh idea bahut innovative hai",
    "Aaj bahut productive din raha",
    "Yeh project bahut interesting hai, kuch naya seekha",
    "Aaj subah jaldi uthke exercise ki",
    "Mere college ke dost bahut achhe hain",
    "Yeh place bahut peaceful hai, yahan aana chahiye",
    "Aaj bahut achha khana khaya",
    "Mujhe photography bahut pasand hai",
    "Yeh series bahut engaging hai, ek baar zaroor dekho",
    "Aaj bahut kuch seekha new technology ke baare mein",
    "Mere mentor bahut helpful hain",
    "Yeh startup idea bahut promising lag raha hai",
    "Aaj team ne bahut achha kaam kiya",
    "Mujhe hiking bahut pasand hai",
    "Yeh article bahut well-written hai",
    "Aaj bahut productive meeting hui",
    "Mere dost ne bahut achha gift diya",
    "Yeh workshop bahut informative thi",
    "Aaj bahut achha din raha overall",
    "Mujhe cooking mein interest hai",
    "Yeh technology ka future bahut bright hai",
    "Aaj bahut kuch accomplish kiya",
    "Mere classmates bahut cooperative hain",
    "Yeh initiative bahut achha hai community ke liye",
    "Aaj bahut relaxing evening thi",
    "Mujhe reading bahut pasand hai",
    "Yeh collaboration bahut fruitful rahi",
    "Aaj bahut achha weather hai bahar",
    "Mere bhai ki shaadi hai is mahine",
    "Yeh festival mein bahut maza aata hai",
    "Aaj naya phone liya, bahut excited hoon",
    "Mujhe swimming bahut pasand hai",
    "Yeh lecture bahut informative tha",
    "Aaj bahut achha workout kiya",
    "Mere dost ne bahut achha advice diya",
    "Yeh place bahut beautiful hai, zaroor jao",
    "Aaj bahut kuch naya seekha",
    "Mujhe drawing bahut pasand hai",
    "Yeh initiative bahut helpful hai students ke liye",
    "Aaj bahut achha time spend kiya family ke saath",
    "Mere teacher ne bahut achha samjhaya",
    "Yeh book bahut motivating hai, padho zaroor",
    "Aaj bahut achha feel ho raha hai",
    "Mujhe nature photography bahut pasand hai",
    "Yeh project successfully complete ho gaya",
    "Aaj bahut productive session tha",
    "Mere dost bahut talented hain",
    "Yeh course ne bahut kuch sikhaya mujhe",
    "Aaj bahut achha experience raha",
    "Mujhe travelling bahut pasand hai",
    "Yeh idea implement karna chahiye",
    "Aaj bahut achha din guzra",
    "Mere parents ne bahut support kiya",
    "Yeh technology bahut helpful hai",
]


def generate_dataset(output_path: str = "data/hinglish_hate_speech.csv", augment: bool = True):
    """
    Builds the dataset and saves as CSV.
    Augments with word-swap to increase diversity.
    """
    hate_texts = HATE_SAMPLES.copy()
    non_hate_texts = NON_HATE_SAMPLES.copy()

    if augment:
        def augment_sentence(sentence):
            words = sentence.split()
            if len(words) > 4:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
            return " ".join(words)

        # Augment to balance classes
        n_hate = len(hate_texts)
        n_non_hate = len(non_hate_texts)
        target = max(n_hate, n_non_hate) + 100

        extra_hate = [augment_sentence(s) for s in random.choices(HATE_SAMPLES, k=target - n_hate)]
        extra_non_hate = [augment_sentence(s) for s in random.choices(NON_HATE_SAMPLES, k=target - n_non_hate)]
        hate_texts += extra_hate
        non_hate_texts += extra_non_hate

    texts = hate_texts + non_hate_texts
    labels = [1] * len(hate_texts) + [0] * len(non_hate_texts)

    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[dataset] Saved {len(df)} samples → {df['label'].sum()} hate, "
          f"{(df['label']==0).sum()} non-hate")
    return df


if __name__ == "__main__":
    generate_dataset()
