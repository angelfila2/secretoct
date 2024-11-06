// Function to toggle the chat visibility
function toggleChat() {
    const chatContainer = document.getElementById("chat-container");
    if (chatContainer.style.display === "none" || chatContainer.style.display === "") {
        chatContainer.style.display = "flex"; // Show the chat container
    } else {
        chatContainer.style.display = "none"; // Hide the chat container
    }
}

// Function to send a message and get a response from the backend
async function sendMessage() {
    const userMessageInput = document.getElementById("userInput");
    const userMessage = userMessageInput.value;
    if (!userMessage.trim()) return; // Prevent sending empty messages

    // Display the user's message in the chat box
    const userMessageElem = document.createElement("p");
    userMessageElem.classList.add("user-message");
    userMessageElem.innerText = userMessage;
    document.getElementById("chatbox").appendChild(userMessageElem);

    // Clear the input field and scroll to the latest message
    userMessageInput.value = "";
    document.getElementById("chatbox").scrollTop = document.getElementById("chatbox").scrollHeight;

    try {
        // Send the user's message to the backend via POST request
        const response = await fetch("/get_response", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: userMessage })
        });

        // Parse the JSON response
        const data = await response.json();

        // Display the bot's response in the chat box
        const botMessageElem = document.createElement("p");
        botMessageElem.classList.add("bot-message");
        botMessageElem.innerText = data.response;
        document.getElementById("chatbox").appendChild(botMessageElem);

        // Scroll to the latest message
        document.getElementById("chatbox").scrollTop = document.getElementById("chatbox").scrollHeight;
    } catch (error) {
        console.error("Error sending message:", error);
        // Optionally, you could display an error message in the chat
    }
}

// Add event listener to detect "Enter" keypress in the input field
document.getElementById("userInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();  // Prevent newline in the input field
        sendMessage();           // Call sendMessage function
    }
});
