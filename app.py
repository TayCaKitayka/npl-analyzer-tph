import os
import torch
from transformers import pipeline
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, CallbackContext

TOKEN = "7861972524:AAHGGP0KyK4SFNxoECnOwtmskr3q6Qw5758"

MODEL_PATH =  r"C:\Users\TayCa\Desktop\last\model_training"

classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)

label_map = {
    0: "sadness",
    1: "anger",
    2: "joy",
    3: "neutral"
}

async def analyze_sentiment(update: Update, context: CallbackContext):
    text = update.message.text
    result = classifier(text)[0]
    label_index = int(result["label"].split("_")[-1])
    sentiment = label_map[label_index]
    response = f"*Mood* {sentiment}"
    await update.message.reply_text(response, parse_mode="Markdown")


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("hey, im a bot :D")


def main():
    app = Application.builder().token(TOKEN).build()


    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_sentiment))

    print("Start bot")
    app.run_polling()


if __name__ == "__main__":
    main()
