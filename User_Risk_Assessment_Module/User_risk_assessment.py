"""This code is by nitin"""

""""
Citation - Title : Financial risk tolerance revisited, the development of a risk assessment instrument
Authors : John Grable, Ruth H. Lytton
Journal : Financial Services Review 8 (1999) 163–181
"""

import streamlit as st
 
# Define the exchange rate from USD to INR
USD_TO_INR_RATE = 100  # Updated as per user request
 
# Define the questions, options, and scoring
QUESTIONS = [
    {
        "question": "In general, how would your best friend describe you as a risk taker?",
        "options": {
            "a": {"text": "A real gambler", "score": 4},
            "b": {"text": "Willing to take risks after completing adequate research", "score": 3},
            "c": {"text": "Cautious", "score": 2},
            "d": {"text": "A real risk avoider", "score": 1},
        },
    },
    {
        "question": "You are on a TV game show and can choose one of the following. Which would you take?",
        "options": {
            "a": {"text": f"$1,000 in cash", "score": 1},
            "b": {"text": "A 50% chance at winning $5,000", "score": 2},
            "c": {"text": "A 25% chance at winning $10,000", "score": 3},
            "d": {"text": "A 5% chance at winning $100,000", "score": 4},
        },
    },
    {
        "question": "You have just finished saving for a “once-in-a-lifetime” vacation. Three weeks before you plan to leave, you lose your job. You would:",
        "options": {
            "a": {"text": "Cancel the vacation", "score": 1},
            "b": {"text": "Take a much more modest vacation", "score": 2},
            "c": {"text": "Go as scheduled, reasoning that you need the time to prepare for a job search", "score": 3},
            "d": {"text": "Extend your vacation, because this might be your last chance to go first-class", "score": 4},
        },
    },
    {
        "question": "How would you respond to the following statement? “It’s hard for me to pass up a bargain.”",
        "options": {
            "a": {"text": "Very true", "score": 1},
            "b": {"text": "Sometimes true", "score": 2},
            "c": {"text": "Not at all true", "score": 3},
        },
    },
    {
        "question": "If you unexpectedly received $20,000 to invest, what would you do?",
        "options": {
            "a": {"text": "Deposit it in a bank account, money market account, or an insured CD", "score": 1},
            "b": {"text": "Invest it in safe high quality bonds or bond mutual funds", "score": 2},
            "c": {"text": "Invest it in stocks or stock mutual funds", "score": 3},
        },
    },
    {
        "question": "In terms of experience, how comfortable are you investing in stocks or stock mutual funds?",
        "options": {
            "a": {"text": "Not at all comfortable", "score": 1},
            "b": {"text": "Somewhat comfortable", "score": 2},
            "c": {"text": "Very comfortable", "score": 3},
        },
    },
    {
        "question": "Which situation would make you the happiest?",
        "options": {
            "a": {"text": "You win $50,000 in a publisher’s contest", "score": 2},
            "b": {"text": "You inherit $50,000 from a rich relative", "score": 1},
            "c": {"text": "You earn $50,000 by risking $1,000 in the options market", "score": 3},
            "d": {"text": "Any of the above—after all, you’re happy with the ₹5,000,000", "score": 1},  # Note: Converted $50,000 to INR
        },
    },
    {
        "question": "When you think of the word “risk” which of the following words comes to mind first?",
        "options": {
            "a": {"text": "Loss", "score": 1},
            "b": {"text": "Uncertainty", "score": 2},
            "c": {"text": "Opportunity", "score": 3},
            "d": {"text": "Thrill", "score": 4},
        },
    },
    {
        "question": "You inherit a mortgage-free house worth $80,000. The house is in a nice neighborhood, and you believe that it should increase in value faster than inflation. Unfortunately, the house needs repairs. If rented today, the house would bring in $600 monthly, but if updates and repairs were made, the house would rent for $800 per month. To finance the repairs you’ll need to take out a mortgage on the property. You would:",
        "options": {
            "a": {"text": "Sell the house", "score": 1},
            "b": {"text": "Rent the house as is", "score": 2},
            "c": {"text": "Remodel and update the house, and then rent it", "score": 3},
        },
    },
    {
        "question": "In your opinion, is it more important to be protected from rising consumer prices (inflation) or to maintain the safety of your money from loss or theft?",
        "options": {
            "a": {"text": "Much more important to secure the safety of my money", "score": 1},
            "b": {"text": "Much more important to be protected from rising prices (inflation)", "score": 3},
        },
    },
    {
        "question": "You’ve just taken a job at a small fast growing company. After your first year you are offered the following bonus choices. Which one would you choose?",
        "options": {
            "a": {"text": "A five year employment contract", "score": 1},
            "b": {"text": "A ₹2,500,000 bonus", "score": 2},  # Converted $25,000 to INR
            "c": {"text": "Stock in the company currently worth ₹2,500,000 with the hope of selling out later at a large profit", "score": 3},
        },
    },
    {
        "question": "Some experts are predicting prices of assets such as gold, jewels, collectibles, and real estate (hard assets) to increase in value; bond prices may fall, however, experts tend to agree that government bonds are relatively safe. Most of your investment assets are now in high interest government bonds. What would you do?",
        "options": {
            "a": {"text": "Hold the bonds", "score": 1},
            "b": {"text": "Sell the bonds, put half the proceeds into money market accounts, and the other half into hard assets", "score": 2},
            "c": {"text": "Sell the bonds and put the total proceeds into hard assets", "score": 3},
            "d": {"text": "Sell the bonds, put all the money into hard assets, and borrow additional money to buy more", "score": 4},
        },
    },
    {
        "question": "Assume you are going to buy a home in the next few weeks. Your strategy would probably be:",
        "options": {
            "a": {"text": "To buy an affordable house where you can make monthly payments comfortably", "score": 1},
            "b": {"text": "To stretch a bit financially to buy the house you really want", "score": 2},
            "c": {"text": "To buy the most expensive house you can qualify for", "score": 3},
            "d": {"text": "To borrow money from friends and relatives so you can qualify for a bigger mortgage", "score": 4},
        },
    },
    {
        "question": "Given the best and worst case returns of the four investment choices below, which would you prefer?",
        "options": {
            "a": {"text": "₹20,000 gain best case; ₹0 gain/loss worst case", "score": 1},
            "b": {"text": "₹80,000 gain best case; ₹20,000 loss worst case", "score": 2},
            "c": {"text": "₹260,000 gain best case; ₹80,000 loss worst case", "score": 3},
            "d": {"text": "₹480,000 gain best case; ₹240,000 loss worst case", "score": 4},
        },
    },
    {
        "question": "Assume that you are applying for a mortgage. Interest rates have been coming down over the past few months. There’s the possibility that this trend will continue. But some economists are predicting rates to increase. You have the option of locking in your mortgage interest rate or letting it float. If you lock in, you will get the current rate, even if interest rates go up. If the rates go down, you’ll have to settle for the higher locked in rate. You plan to live in the house for at least three years. What would you do?",
        "options": {
            "a": {"text": "Definitely lock in the interest rate", "score": 1},
            "b": {"text": "Probably lock in the interest rate", "score": 2},
            "c": {"text": "Probably let the interest rate float", "score": 2},
            "d": {"text": "Definitely let the interest rate float", "score": 3},
        },
    },
    {
        "question": "In addition to whatever you own, you have been given ₹100,000. You are now asked to choose between:",
        "options": {
            "a": {"text": "A sure gain of ₹50,000", "score": 1},
            "b": {"text": "A 50% chance to gain ₹100,000 and a 50% chance to gain nothing", "score": 3},
        },
    },
    {
        "question": "In addition to whatever you own, you have been given ₹200,000. You are now asked to choose between:",
        "options": {
            "a": {"text": "A sure loss of ₹50,000", "score": 1},
            "b": {"text": "A 50% chance to lose ₹100,000 and a 50% chance to lose nothing", "score": 3},
        },
    },
    {
        "question": "Suppose a relative left you an inheritance of ₹10,000,000, stipulating in the will that you invest ALL the money in ONE of the following choices. Which one would you select?",
        "options": {
            "a": {"text": "A savings account or money market mutual fund", "score": 1},
            "b": {"text": "A mutual fund that owns stocks and bonds", "score": 2},
            "c": {"text": "A portfolio of 15 common stocks", "score": 3},
            "d": {"text": "Commodities like gold, silver, and oil", "score": 4},
        },
    },
    {
        "question": "If you had to invest ₹20,000, which of the following investment choices would you find most appealing?",
        "options": {
            "a": {"text": "60% in low-risk investments 30% in medium-risk investments 10% in high-risk investments", "score": 1},
            "b": {"text": "30% in low-risk investments 40% in medium-risk investments 30% in high-risk investments", "score": 2},
            "c": {"text": "10% in low-risk investments 40% in medium-risk investments 50% in high-risk investments", "score": 3},
        },
    },
    {
        "question": "Your trusted friend and neighbor, an experienced geologist, is putting together a group of investors to fund an exploratory gold mining venture. The venture could pay back 50 to 100 times the investment if successful. If the mine is a bust, the entire investment is worthless. Your friend estimates the chance of success is only 20%. If you had the money, how much would you invest?",
        "options": {
            "a": {"text": "Nothing", "score": 1},
            "b": {"text": "One month’s salary", "score": 2},
            "c": {"text": "Three month’s salary", "score": 3},
            "d": {"text": "Six month’s salary", "score": 4},
        },
    },
]
 
def convert_usd_to_inr(usd):
    return usd * USD_TO_INR_RATE
 
def main():
    st.title("📊 User Risk Assessment for Indian Retail Traders")
 
    # Initialize session state variables
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = [None] * len(QUESTIONS)
 
    current_q = st.session_state.current_question
    question = QUESTIONS[current_q]
 
    st.subheader(f"Question {current_q + 1} of {len(QUESTIONS)}")
    st.write(question["question"])
 
    # Prepare options with converted currency if needed
    options = {}
    for key, option in question["options"].items():
        text = option["text"]
        # Detect and convert currency in the option text
        if "$" in text:
            parts = text.split("$")
            amount_part = parts[1].split()[0].replace(",", "")  # Get the number after $
            try:
                amount_usd = float(amount_part)
                amount_inr = convert_usd_to_inr(amount_usd)
                text = text.replace(f"${amount_usd:,.0f}", f"₹{amount_inr:,.0f}")
            except ValueError:
                pass  # If conversion fails, keep the original text
        elif "₹" in text:
            # Ensure proper formatting if INR is already present
            pass  # Assuming INR amounts are already correctly formatted
        options[key] = text
 
    # Display radio buttons for options
    selected_option = st.radio("Select an option:", list(options.keys()), format_func=lambda x: f"{x}. {options[x]}",
                               index=list(options.keys()).index(st.session_state.responses[current_q]) if st.session_state.responses[current_q] in options else 0)
 
    # Save the response
    st.session_state.responses[current_q] = selected_option
 
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
 
    with col1:
        if st.button("Previous"):
            if st.session_state.current_question > 0:
                st.session_state.current_question -= 1
 
    with col3:
        if st.button("Next"):
            if st.session_state.current_question < len(QUESTIONS) - 1:
                st.session_state.current_question += 1
 
    # If on the last question, show the result
    if current_q == len(QUESTIONS) - 1:
        if all(response is not None for response in st.session_state.responses):
            total_score = 0
            for idx, response in enumerate(st.session_state.responses):
                # Some questions might have incomplete options
                try:
                    score = QUESTIONS[idx]["options"][response]["score"]
                    total_score += score
                except KeyError:
                    st.error(f"Invalid response for question {idx + 1}.")
                    return
 
            st.success(f"Your Total Risk Score is: {total_score}")
 
            # Provide a simple risk profile based on total score
            max_score = sum([max(q["options"][opt]["score"] for opt in q["options"]) for q in QUESTIONS])
            risk_percentage = (total_score / max_score) * 100
 
            if risk_percentage >= 75:
                profile = "High Risk Taker"
            elif risk_percentage >= 50:
                profile = "Moderate Risk Taker"
            else:
                profile = "Low Risk Taker"
 
            st.info(f"Your Risk Profile: **{profile}**")
        else:
            st.warning("Please answer all questions to see your risk assessment.")
 
    # Optionally, provide a reset button
    if st.button("Reset Assessment"):
        st.session_state.current_question = 0
        st.session_state.responses = [None] * len(QUESTIONS)
        st.experimental_rerun()
 
if __name__ == "__main__":
    main()
