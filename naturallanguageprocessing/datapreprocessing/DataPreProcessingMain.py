from KoreanDataCleaning import KoreanDataCleaning
from KoreanTokenization import KoreanTokenization
from KoreanStopwordsRemoval import KoreanStopwordsRemoval
from KoreanLemmatization import KoreanLemmatization
from KoreanEncoding import KoreanEncoding
from KoreanNormalization import KoreanNormalization

def main():
    
    korean_text = "이것은 한국어 테스트 문장 입니다."
    
    # step 1 datacleaning 
    cleaner = KoreanDataCleaning(korean_text)
    cleaner.remove_special_characters()
    cleaner.remove_html_tags()
    cleaner.to_lowercase()
    cleanned_text = cleaner.get_text()
    print(f"KoreanDataCleaning => {cleanned_text}")
    
    # step 2 tokenization
    tokenizer = KoreanTokenization(cleanned_text)
    korean_tokens = tokenizer.tokenize()
    print(f"KoreanTokenization => {korean_tokens}")
    
    #step 3 stopwordsRemoval
    stopwords_removal = KoreanStopwordsRemoval(korean_tokens)
    filterd_korean_tokens = stopwords_removal.remove_stopwords()
    print(f"KoreanStopwordsRemoval => {filterd_korean_tokens}")

    #step 4 lemmatization
    lemmatizer= KoreanLemmatization(filterd_korean_tokens)
    lemmatized_korean_tokens = lemmatizer.apply_lemmatization()
    print(f"KoreanLemmatization => {lemmatized_korean_tokens}")
    
    #step 5 encoding
    encoder = KoreanEncoding(lemmatized_korean_tokens)
    encoded_korean_tokens = encoder.encode()
    print(f"KoreanEncoding => {encoded_korean_tokens}")

    #step 6 normalization
    normalizer = KoreanNormalization(encoded_korean_tokens)
    normalized_data = normalizer.normalize(max_length=10)
    print(f"KoreanNormalization => {normalized_data}")

if __name__ == "__main__":
    main()