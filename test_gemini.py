from google import genai; client = genai.Client(); print(client.models.generate_content(model="gemini-2.5-flash", contents="Say Flash is working!").text)
