
import os
import sys
import time
import telebot
import logging
from telebot import types

from smart_pot import config
from smart_pot.image_processing import ImageProcessing
from model.resnet_bottleneck import ResNetBottleneckModel
from model.densenet_bottleneck import DenseNetBottleneckModel
from model.nasnet_bottleneck import NASNetBottleneckModel
from model.vgg_bottleneck_fine_tuned import VGGBottleneckModelFineTuned
from model.vgg_bottleneck import VGGBottleneckModel

logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)

bot = telebot.TeleBot(config.TOKEN)
img_prep = ImageProcessing(model=VGGBottleneckModel, dims=[2048, 512], find_pca=False)
executable = sys.executable


def _find_similar(img_path):
    similar_imgs, links = img_prep.main(img_path)
    return similar_imgs, links


def _write_logs(image=None, similar=None, response=None):
    log_file_name = os.path.join("logs", "smart_pot_{0}.csv".format(VGGBottleneckModel.get_name()))
    with open(log_file_name, "a") as file:
        if image:
            file.write("query:{0}\nresults:".format(image))
            log = ""
            for s in similar:
                log += s
                log += ","
            log += "\n"
            file.write(log)
        elif response:
            file.write("response:{0}\n".format(response))


@bot.message_handler(commands=["start"])
def send_welcome(message): 
    bot.send_message(
        chat_id=message.chat.id,
        text=config.COMMANDS["start"]
        )


@bot.message_handler(content_types=["text"])
def send_welcome(message): 
    bot.send_message(
        chat_id=message.chat.id,
        text=config.COMMANDS["gen"]
        )


@bot.message_handler(content_types=["photo"]) 
def send_similar_images(message):
    try:
        raw = message.photo[2].file_id
    except:
        raw = message.photo[1].file_id
    file_name = raw + ".jpg"
    img_path = os.path.join("./bot_images", file_name)
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(img_path, "wb") as new_file:
        new_file.write(downloaded_file)

    bot.send_message(
        chat_id=message.chat.id,
        text=config.COMMANDS["search"]
    )

    img_path = os.path.join("bot_images", file_name)
    imgs, links = _find_similar(img_path)

    for n, img, link in zip(config.nums, imgs, links):
        try:
            file = open(os.path.join("..", "media", img), "rb")
            bot.send_photo(message.chat.id, file)
            bot.send_message(message.chat.id, "№{0} {1}".format(n, link))
        except:
            print("Не смог отправить фото {0}".format(img))
            pass

    _write_logs(image=file_name, similar=imgs)

    keyboard = types.InlineKeyboardMarkup(row_width=2)
    keyboard.add(*[
            types.InlineKeyboardButton(text=name, callback_data=name)
            for name in config.ANSWERS.values()
        ])
    bot.send_message(
        chat_id=message.chat.id,
        text=config.COMMANDS["poll"],
        reply_markup=keyboard
        )


@bot.callback_query_handler(func=lambda c: True)
def inline(c):
    if c.data == config.ANSWERS["pos"]:
        config.POS += 1
        print("POS = {0}".format(config.POS))
        bot.edit_message_text(
            chat_id=c.message.chat.id,
            message_id=c.message.message_id,
            text=config.COMMANDS["pos"],
            parse_mode="Markdown"
            )

        keyboard = types.InlineKeyboardMarkup(row_width=3)
        keyboard.add(*[
            types.InlineKeyboardButton(text=name, callback_data=name)
            for name in config.nums
        ])
        keyboard.add(types.InlineKeyboardButton(text=config.COMMANDS["next"], callback_data=config.COMMANDS["next"]))
        bot.send_message(
            chat_id=c.message.chat.id,
            text=config.COMMANDS["resp"],
            reply_markup=keyboard
        )
        _write_logs(response=c.data)

    elif c.data == config.ANSWERS["neg"]:
        config.NEG += 1
        print("NEG = {0}".format(config.NEG))
        bot.edit_message_text(
            chat_id=c.message.chat.id,
            message_id=c.message.message_id,
            text=config.COMMANDS["neg"],
            parse_mode="Markdown"
            )
        _write_logs(response=c.data)

    elif c.data == config.ANSWERS["unk"]:
        config.UNK += 1
        print("UNK = {0}".format(config.UNK))
        bot.edit_message_text(
            chat_id=c.message.chat.id,
            message_id=c.message.message_id,
            text=config.COMMANDS["unk"],
            parse_mode="Markdown"
            )
        _write_logs(response=c.data)

    elif c.data in config.nums:
        _write_logs(response=c.data)

        bot.edit_message_text(
            chat_id=c.message.chat.id,
            message_id=c.message.message_id,
            text=config.COMMANDS["chosen"].format(c.data),
            parse_mode="Markdown"
        )

        keyboard = types.InlineKeyboardMarkup(row_width=3)
        keyboard.add(*[
            types.InlineKeyboardButton(text=name, callback_data=name)
            for name in config.nums
        ])
        keyboard.add(types.InlineKeyboardButton(text=config.COMMANDS["next"], callback_data=config.COMMANDS["next"]))
        bot.send_message(
            chat_id=c.message.chat.id,
            text=config.COMMANDS["resp"],
            reply_markup=keyboard
        )

    elif c.data == config.COMMANDS["next"]:
        bot.edit_message_text(
            chat_id=c.message.chat.id,
            message_id=c.message.message_id,
            text=config.COMMANDS["num"] + "\n" + config.COMMANDS["gen"],
            parse_mode="Markdown"
        )


if __name__ == "__main__":
    try:
        bot.polling(none_stop=True, interval=0, timeout=30)
    except Exception as err:
        print(err)
        time.sleep(3)
        os.execl(executable, executable, "./bot.py")
