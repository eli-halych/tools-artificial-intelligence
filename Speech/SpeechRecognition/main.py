import webbrowser
import string
import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Hi, how can I help?[Weather]")
    audio = r.listen(source)

if "weather" in r.recognize_google(audio):
    r2 = sr.Recognizer()
    url2 = 'https://www.yr.no/place/Poland/Masovia/'
    with sr.Microphone() as source:
        print("Please, say a city in Poland")
        audio2 = r2.listen(source)




        try:
            print("Google Speech Recognition thinks you said " + r2.recognize_google(audio2))
            webbrowser.open_new(url2 + r2.recognize_google(audio2))
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

