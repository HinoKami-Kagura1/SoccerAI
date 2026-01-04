from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
import re

load_dotenv()


class SoccerFilter:

    def __init__(self):
   
        self.soccer_keywords = {
            'soccer', 'football', 'goal', 'penalty', 'offside', 'fifa', 'uefa',
            'premier', 'league', 'champions', 'world', 'cup', 'match', 'player',
            'team', 'manager', 'coach', 'referee', 'transfer', 'formation',
            'stadium', 'corner', 'kick', 'var', 'dribble', 'pass', 'shot', 'save',
            'keeper', 'defender', 'midfielder', 'forward', 'striker', 'winger',
            'captain', 'substitute', 'fixture', 'derby', 'tournament',

            'goals', 'scored', 'score', 'statistics', 'stats', 'stat', 
            'assists', 'titles', 'trophies', 'records', 'ranking', 'career'

            'real madrid', 'barcelona', 'manchester united', 'manchester city', 
            'liverpool', 'bayern', 'munich', 'psg', 'chelsea', 'arsenal', 
            'ac milan', 'inter', 'juventus', 'ajax', 'dortmund', 'atletico',
            'benfica', 'porto', 'tottenham', 'spurs'
        }

        self.blocked_topics = [
            'weather', 'cooking', 'recipe', 'movie', 'music', 'politics',
            'stock', 'business', 'math', 'science', 'programming', 'bake',
            'food', 'restaurant', 'car', 'travel', 'shopping', 'phone',
            'computer', 'video game', 'tv show', 'netflix'
        ]

        self.other_sports = [
            'basketball', 'baseball', 'tennis', 'cricket', 'golf', 'hockey',
            'rugby', 'volleyball', 'boxing', 'mma', 'ufc', 'nfl', 'nba'
        ]

    def clean_question(self, question):
        question = re.sub(r'[^a-z\s]', ' ', question.lower())
        return question

    def is_soccer_related(self, question):

        cleaned = self.clean_question(question)
        words = cleaned.split()
        question_lower = question.lower()
        
        for topic in self.blocked_topics + self.other_sports:
            if topic in cleaned:
                return False, f"Contains blocked topic: {topic}"

        soccer_words = [word for word in words if word in self.soccer_keywords]
        if soccer_words:
            return True, f"Found soccer terms: {', '.join(soccer_words)}"
        
        stat_patterns = ['goals', 'assists', 'trophies', 'records', 'statistics', 'scored', 'titles']
        has_stats = any(pattern in question_lower for pattern in stat_patterns)
        
        if has_stats:
            if len(words) >= 4:
                return True, "Player statistics query detected"
        
        player_id_patterns = ['which player', 'who is', 'best player', 'top player', 'greatest player']
        if any(phrase in question_lower for phrase in player_id_patterns):
            return True, "Player identification query"
        
        team_indicators = ['team', 'club', 'transfer', 'signed', 'plays for', 'contract']
        if any(indicator in question_lower for indicator in team_indicators):

            if not any(sport in question_lower for sport in self.other_sports):
                return True, "Team-player relationship query"
        
        comparison_patterns = ['vs', 'versus', 'compare', 'better than', 'rivalry']
        if any(pattern in question_lower for pattern in comparison_patterns):

            if len(words) >= 5: 
                return True, "Comparison query"
        
        history_patterns = ['retired', 'legend', 'legendary', 'history', 'career', 'played for']
        if any(pattern in question_lower for pattern in history_patterns):
            if len(words) >= 4:
                return True, "Player history query"
        
        question_words = ['how', 'what', 'when', 'where', 'why', 'which', 'who']
        has_question_word = any(word in question_words for word in words[:3]) 
        has_player_context = any(word in ['player', 'players', 'footballer'] for word in words)
        
        if has_question_word and has_player_context:
            return True, "General player query"
        
        return False, "No soccer context detected"


def main():

    filter = SoccerFilter()

    print("Welcome to SoccerGPT!")
    print("I can answer any soccer/football related questions!")
    print("Examples: 'Who won the World Cup?', 'Explain offside rule'")
    print("Type 'exit' to quit")

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )

    conversation = ConversationChain(
        llm=llm,
        memory=ConversationSummaryMemory(llm=llm),
        verbose=False
    )

    while True:

        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nSoccerGPT: Thanks for chatting about soccer!")
                break

            if not user_input:
                print("SoccerGPT: Please ask a question!")
                continue

            is_soccer, reason = filter.is_soccer_related(user_input)

            if not is_soccer:
                print(f"SoccerGPT: I only answer soccer/football questions! ({reason})")
                print("Try asking about: players, teams, matches, rules, or tactics")
                continue

            print("SoccerGPT: Thinking...")
            response = conversation.predict(input=user_input)
            print(f"SoccerGPT: {response}")

        except KeyboardInterrupt:
            print("\n\nSoccerGPT: Session ended. Come back for more soccer talk!")
            break

        except Exception as e:
            print(f"SoccerGPT: Oops, an error occurred: {str(e)}")
            print("SoccerGPT: Please try again in a few moments.")


if __name__ == "__main__":
    main()