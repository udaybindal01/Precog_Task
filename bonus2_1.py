import google.generativeai as ai


ai.configure(api_key="AIzaSyBsROOsRnI1JopbvCzM2-FpkSre0lFzaXo")

def Check(word1,word2):
    
    query = f"Are {word1} and {word2} similar?  To calculate the similarity . convert each word to vector using word2vec and then take mean of the vectors to get a vector for that sentence . After getting the vectors for both the sentences calculate cosine similarity. Just give me the similarity score"
    model = ai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    # print(response)
    print(word1,word2,response.text)

word1 = "This is a cat"
word2 = "there u go"
Check(word1,word2)