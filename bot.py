from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging


PROXY = {'proxy_url': 'socks5://t1.learn.python.ru:1080',

'urllib3_proxy_kwargs': {'username': 'learn', 'password': 'python'}}

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    filename='bot.log'
                    )

teleg_log=logging.getLogger('telegram.ext.updater')

def greet_user(bot, update):
    text = 'Вызван /start'
    logging.info(text)
    update.message.reply_text(text)

def talk_to_me(bot,update):
    user_text = "Привет {}! Ты написал: {}".format(update.message.chat.first_name, update.message.text) 
    logging.info("User: %s, Chat id: %s, Message: %s", update.message.chat.username, 
                    update.message.chat.id, update.message.text)
    update.message.reply_text(user_text)


def main():
    mybot = Updater("650028034:AAH-dqfS_nkiSA7DTLzpSihJKRz8aX3eGJg", request_kwargs=PROXY)
   

    logging.info('Обожемой, бот кому-то понадобился!')

    dp = mybot.dispatcher
    dp.add_handler(CommandHandler('start', greet_user))
    dp.add_handler(MessageHandler(Filters.text, talk_to_me))

    mybot.start_polling()
    mybot.idle()


main()