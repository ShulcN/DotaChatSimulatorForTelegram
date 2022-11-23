"""
This is a echo bot.
It echoes any incoming text messages.
"""
import random
import logging
import torch
from bert_classifier import BertClassifier
from deeppavlov import build_model
import pandas as pd
from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = '5817022810:AAHr4f8l_a0xZqqafcM3vAByng3EPH3hQ18'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

global mean
global answerer
global model
global texts
global items
global prev_ans
global repeat
global prev_ph


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    prev_ans[message.chat.id] = []
    prev_ph[message.chat.id] = []
    await message.reply("Привет! Моя главная задача погрузить тебя в мир Dota 2. Желаю преятно провести время😉")


@dp.message_handler()
async def echo(message: types.Message):
    if not prev_ans[message.chat.id]:
        prev_ans[message.chat.id] = []
        prev_ph[message.chat.id] = []
    if message.text == 'Да' or message.text == 'да':
        await message.answer('провода')
        return
    if message.text == 'нет' or message.text == 'Нет':
        await message.answer('обезьяны ответ')
        return

    n = model.predict(message.text)
    ans = ''
    global texts_temp
    if len(texts[n])>5:
        texts_temp = random.choices(texts[n], k=5)
    else:
        texts_temp = texts[n]
    max_val = 0
    while texts_temp:
        temp = answerer([texts_temp.pop()], [message.text])
        print(temp)
        if temp[2][0]>max_val and temp[0][0]:
            max_val = temp[2][0]
            ans = temp[0][0]
            if max_val == 1.0:
                break

    if not ans:
        text = random.choice(
            texts[random.choice([0, 1, 2, 3, 4, 5, 2, 3])])
        ans = answerer([text], [message.text])[0][0]
    if not ans:
        ans = random.choice(items)
    if ans in prev_ans[message.chat.id]:
        if ans not in items:
            ans = random.choice(repeat)
        while (ans in repeat) and (ans in prev_ans[message.chat.id]):
            ans = random.choice(repeat)
        while (ans in items) and (ans in prev_ans[message.chat.id]):
            if ans == 'image':
                break
            ans = random.choice(items)
    if ans == 'image':
        ans = random.choice(photos)
        while ans in prev_ph[message.chat.id]:
            ans = random.choice(photos)
        prev_ph[message.chat.id].append(ans)
        photo = open(ans, "rb")
        await bot.send_photo(chat_id=message.chat.id, photo=photo)
        return
    prev_ans[message.chat.id].append(ans)
    if len(prev_ans) > 15:
        prev_ans.pop(0)
    print(message.text + ' |||| тема: ' + mean[n] + ' |||| ответ:' + ans)
    await message.answer(ans)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ = torch.load('bert_3.pt', map_location=device)
    model_.load_state_dict(torch.load('state_dict_1.pt',
                                      map_location=device))
    model_.eval()
    model = BertClassifier(model_, n_classes=6)
    answerer = build_model('squad_ru_bert', download=True)
    data = pd.read_csv('Dota_talk - Copy.csv')
    texts = dict()
    for d in data['theme'].unique():
        texts[d] = []
    for n, t in zip(data['theme'], data['text']):
        texts[n].append(t)
    repeat = ['я уже говорил', 'мне повторить?', 'у тебя память как у рыбки? я уже говорил', 'я уже писал',
              'заколебал, я уже отвечал', 'я уже отвечал', 'сколько раз мне повторять?']
    prev_ans = dict()
    prev_ph = dict()
    photos = ['ph/6b481a4b463c47e421eda249d837b423.jpg', 'ph/94daa326177cff8b854fc50bfc2ab9f2.jpg',
              'ph/ab23171d63e68004a0e3957d5b3b0b06.jpg', 'bot/ph/bcd785e72da7efc428fda2d547750317.jpg', 'ph/1.jpg',
              'ph/2.jpeg', 'ph/3.jpg', 'ph/4.jpg', 'ph/5.jpg', 'ph/6.jpg', 'ph/7.jpg', 'ph/8.jpg', 'ph/9.jpg']
    mean = {
        0: "родители",
        1: "фанат",
        2: "оскорбления",
        3: "общее",
        4: "политика",
        5: "dota",
        6: "хз что это",
    }
    #items = ['image']
    items = ['что? повтори пж', 'я не пон', 'перестань держать уже член и начни писать нормально',
             'когда тебе в рот чем-то тыкают писать не удобно кнш', 'а ты смешной', 'пж больше не пиши сюда',
             'не лезь в эту тему дура', 'я тебя не понимаю', 'что с тобой не так?', 'не пиши сюда больше', 'пон', 'wtf',
             'мама твоя', 'я учту твое мнение, ахахахахахахах', 'конч', 'вот что я должен ответить?',
             'не выпендривайся', 'помолчи лучше', 'ты в муте, бездарь', 'не пукай', 'пук-пук',
             'хуже тебя только жирков', 'ты мне уже наскучил', 'не интересен ты мне', 'ты мне надоел',
             'я от тебя устал', 'мне надоедо с тобой общаться', 'я тебя не вразумив', 'not understand',
             'image', 'вынь руки из штанов и пиши понятно', 'хватит наяривать, пиши членораздельно']
    executor.start_polling(dp, skip_updates=True)
    print('готов')
