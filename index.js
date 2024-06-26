const puppeteer = require("puppeteer")
const Sentiment = require("sentiment")
const sentiment = new Sentiment()

const SentimentData = async () => {
  const browser = await puppeteer.launch()
  const page = await browser.newPage()
  await page.goto(
    "https://www.hindustantimes.com/world-news/us-news/biden-vs-trump-what-will-be-the-face-off-points-in-first-2024-presidential-debate-101719398569771.html"
  )
  const text = await page.$eval("*", (el) => el.innerText)
  const result = await sentiment.analyze(text)
  console.log(result)
  await browser.close()
}
SentimentData()