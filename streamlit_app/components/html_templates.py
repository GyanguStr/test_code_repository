css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063;
    flex-direction: column; 
    text-align: left; 
}
.chat-message.user-light {
    background-color: #EEF0F4;
}
.chat-message.bot-light {
    background-color: white;
    border: 1px solid #EEF0F4;
    flex-direction: column; 
    text-align: left; 
}
.chat-message .name {
  width: 20%;
}
.chat-message .name {
  max-width: 200px;
  max-height: 78px;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.chat-message .cost {
  color: #A9A9A9;
  font-size: 12px;
}
.mermaid svg {
  height: 1000px;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="answer" style="display: flex; flex-direction: row; align: top">
        <div class="name">
            Bot
        </div>
        <div class="message">{MSG}</div>
    </div>
    <br>
    <div class="cost">
        [{MODEL}] &nbsp;&nbsp;Cost: {COST}$ &nbsp;&nbsp;Tokens used: {TOKENS_USED} (Prompt: {PROMPT}, Completion: {COMPLETION}, Time spent: {TIME} sec)
    </div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="name">
        Human
    </div>    
    <div class="message">{MSG}</div>
</div>
'''

old_bot_template = '''
<div class="chat-message bot">
    <div class="answer" style="display: flex; flex-direction: row; align: top">
        <div class="name">
            Bot
        </div>
        <div class="message">{MSG}</div>
    </div>
    <br>
    <div class="cost">
        [{MODEL}] &nbsp;&nbsp;Cost: {COST}$ &nbsp;&nbsp;Tokens used: {TOKENS_USED} (Prompt: {PROMPT}, Completion: {COMPLETION})
    </div>
</div>
'''

old_light_bot_template = '''
<div class="chat-message bot-light">
    <div class="answer" style="display: flex; flex-direction: row; align: top;">
        <div class="name">
            Bot
        </div>
        <div class="message" style="color: #000000;">{MSG}</div>
    </div>
    <br>
    <div class="cost">
        [{MODEL}] &nbsp;&nbsp;Cost: {COST}$ &nbsp;&nbsp;Tokens used: {TOKENS_USED} (Prompt: {PROMPT}, Completion: {COMPLETION})
    </div>
</div>
'''



light_user_template = '''
<div class="chat-message user-light">
    <div class="name">
        Human
    </div>    
    <div class="message" style="color: #000000;">{MSG}</div>
</div>
'''

hide_bar = """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        visibility:hidden;
        width: 0px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        visibility:hidden;
    }
    </style>
"""

cost_template = '''
<div class="chat-message bot">
    <footer class="cost">
        <span class="model">[{MODEL}]</span>
        <span class="cost-detail">Cost: {COST}$</span>
        <span class="tokens-used">Tokens used: {TOKENS_USED}</span>
        <span class="prompt">Prompt: {PROMPT}</span>
        <span class="completion">Completion: {COMPLETION}</span>
        <span class="time">Time spent: {TIME} sec</span>
    </footer>
</div>
'''