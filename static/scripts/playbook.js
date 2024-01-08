window.addEventListener("load", () => {
    const input = document.getElementById("upload");
    const fileList = document.querySelector(".file-list");
    const submitBtn = document.getElementById("submitBtn");
    const area = document.querySelector(".area");
    const upload_text = document.querySelector(".file-name-text-i");

    // Add event listeners to initial cross icons
    document.querySelectorAll('.file-list .cross-icon').forEach(crossIcon => {
        crossIcon.addEventListener('click', function() {
            this.parentElement.remove();
            updateSubmitButtonState();
        });
    });

    input.addEventListener("change", (e) => {
        const files = e.target.files;
        const allowedFileTypes = ["png", "jpeg", "jpg"]; // Array of allowed file types
    
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const fileName = file.name;
            const fileType = fileName.split(".").pop().toLowerCase();
    
            if (allowedFileTypes.includes(fileType)) { // Check if the file type is allowed
                fileShow(fileName, fileType);
            }
        }
    
        updateSubmitButtonState();
    
        if (e.target.files.length > 0) {
            area.classList.add("hide-background");
        } else {
            area.classList.remove("hide-background");
        }
    });

    const fileShow = (fileName, fileType) => {
        const listItem = document.createElement("li");
        listItem.classList.add("file-item");

        const fileTypeElement = document.createElement("span");
        fileTypeElement.classList.add("file-type");
        fileTypeElement.textContent = fileType;
        listItem.appendChild(fileTypeElement);

        const fileTitleElement = document.createElement("span");
        fileTitleElement.classList.add("file-title");
        fileTitleElement.textContent = fileName;
        listItem.appendChild(fileTitleElement);

        const crossElement = document.createElement("span");
        crossElement.classList.add("cross-icon");
        crossElement.innerHTML = "&#215;";
        listItem.appendChild(crossElement);

        fileList.appendChild(listItem);

        crossElement.addEventListener("click", () => {
            fileList.removeChild(listItem);
            updateSubmitButtonState();
        });
    };

    const updateSubmitButtonState = () => {
        submitBtn.disabled = fileList.childElementCount === 0;
        if (fileList.childElementCount === 0) {
            area.style.backgroundImage = 'url(https://cdn2.iconfinder.com/data/icons/ios-7-icons/50/upload-512.png)';
        } else {
            area.style.backgroundImage = 'none';
        }
        if (fileList.childElementCount > 4) {
            upload_text.style.display = 'none';
        }
    };
});

var chatBlocks = [];
var selectedDropdownItem = null; 

function applyBoxShadow(chatBlock, chatState) {
    if (chatState === "initial") {
        chatBlock.style.boxShadow = "0 4px 8px 0 rgba(0, 0, 0, 0.2)";
    } else if (chatState === "continuous") {
        chatBlock.style.boxShadow = "0 4px 8px 0 rgba(0, 0, 0, 0.3)"; 
    }
}

function addChatBlock(text, sender, chatState,  isTemporary = false) {
    var chatBlock = document.createElement("a");
    chatBlock.href = "#";
    chatBlock.tabIndex = "-1"
    chatBlock.classList.add("list-group-item", "list-group-item-action", "d-flex", "gap-3", "py-3");

    if (isTemporary) {
        chatBlock.id = 'temporary-block';
    }

    var image;
    if (sender === "user") {
        image = document.createElement("img");
        image.src = "../static/images/logo.png";
    } else if (sender === "bot") {
        image = document.createElement("i");
        image.classList.add("fas", "fa-robot", "fa-lg", "rounded-circle", "flex-shrink-0");
        // image.style.paddingTop = "14px";
    }

    image.alt = sender;
    image.width = "30";
    image.height = "30";
    var chatTextContainer = document.createElement("div");
    chatTextContainer.classList.add("d-flex", "gap-2", "w-100", "justify-content-between");
    var chatTextDiv = document.createElement("div");
    var chatText = document.createElement("p");
    chatText.style.paddingTop = "5px";
    chatText.classList.add("mb-0", "opacity-75");
    chatText.id = "chat-text";
    var chatOutput = document.createElement("p");
    chatOutput.classList.add("mb-0", "opacity-75");
    chatOutput.id = "chat-output";
    var formattedText = formatText(text);
    chatOutput.innerHTML = formattedText;
    chatTextDiv.appendChild(chatText);
    chatTextContainer.appendChild(chatTextDiv);
    chatTextDiv.appendChild(chatOutput);
    chatBlock.appendChild(image);
    chatBlock.appendChild(chatTextContainer);
    applyBoxShadow(chatBlock, chatState);
    chatBlocks.push(chatBlock);
    var listGroup = document.getElementById("list-group");
    listGroup.appendChild(chatBlock);
    updateLastChatBlockMargin();
}

function formatText(text) {
    if (typeof text !== 'string') {
        console.error('Invalid input: text must be a string');
        return ''; 
    }
    text = text.replace(/\*\*(.*?)\*\*/g, "<bold>$1</bold>");
    var lines = text.split('\n');
    var newList = [];
    var listItems = [];
    var isList = false;

    lines.forEach(function(line) {
        if (line.match(/^\d+\./) || line.match(/^\s*-\s+(.*)/)) { 
            if (!isList) {
                newList.push('<ul>'); 
                isList = true;
            }
            if (line.startsWith('-')) {
                line = line.replace(/^\s*-\s*/, '');
            } else {
                line = line.substring(line.indexOf(' ') + 1);
            }
            listItems.push('<li>' + line + '</li>');
        } else {
            if (isList) {
                newList.push(listItems.join(''));
                newList.push('</ul>'); 
                listItems = [];
                isList = false;
            }
            newList.push(line); 
        }
    });

    if (isList) {
        newList.push(listItems.join('')); 
        newList.push('</ul>'); 
    }

    var result = newList.join('<br>').replace(/<br><br>/g, '');
    result = result.replace(/<\/li><br><\/ul>/g, '</li></ul>');
    return result;
}


function updateLastChatBlockMargin() {
    if (chatBlocks.length > 0) {
        chatBlocks.forEach(block => {
            block.style.marginBottom = '0';
        });

        chatBlocks[chatBlocks.length - 1].style.marginBottom = '7rem'; 
    }
}

async function handleSubmitInitialPlaybook() {
    const chatInput1 = document.getElementById('chat-input-1');
    const chatInput2 = document.getElementById('chat-input-2');
    const chatText = document.getElementById('chat-text');

    document.getElementById("main-section").style.display = "none";
    document.getElementById("chat-list-section").style.display = "block";
    chatText.textContent = chatInput1.value;
    try {
        const response = await sendRequestPlaybook(chatInput1.value);
        const formattedResponse = formatText(response); 
        const chatOutput = document.getElementById('chat-output');
        chatOutput.innerHTML = formattedResponse;
    } catch (error) {
        console.error(error);
    }
    chatInput2.disabled = false;
}

let imageSent = false;

function sendRequestPlaybook(activeInputValue) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();

        // Append the text input
        formData.append('input', activeInputValue);

        // Append image files only if they haven't been sent before
        if (!imageSent) {
            const imageFiles = document.getElementById('upload').files;
            for (let i = 0; i < imageFiles.length; i++) {
                formData.append('files[]', imageFiles[i]);
            }
            // Set the flag to true as images are being sent this time
            imageSent = true;
        }

        xhr.open('POST', '/playbookchat', true);
        xhr.onload = function() {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                const generatedText = response.generated_text;
                resolve(generatedText);
            } else if (xhr.status === 500) {
                window.location.href = '/error'; 
            } else {
                reject(xhr.statusText);
            }
        };
        xhr.onerror = function() {
            reject(xhr.statusText);
        };

        // Send FormData object
        xhr.send(formData);
    });
}

function checkInputInitialPlaybook() {
    const inputField1 = document.getElementById('chat-input-1');
    const inputField2 = document.getElementById('chat-input-2');
    const submitButton1 = document.querySelector('.btn.chat-btn');
    const submitButton2 = document.getElementById('button2');
    submitButton2.disabled = true;
    inputField2.disabled = true;
    submitButton1.disabled = inputField1.value.trim().length < 2;
}

// Continuous Input Functionality
async function handleSubmitContinuousPlaybook() {
    const chatInput2 = document.getElementById('chat-input-2');
    const submitButton2 = document.getElementById('button2');
    chatInput2.disabled = true;
    submitButton2.disabled = true;
    addChatBlock(chatInput2.value, "user", "continuous");
    addChatBlock('Loading Please Wait...', 'bot', 'continuous', true); 
    
    if (chatInput2.value.trim().length >= 2) {
        
        document.getElementById("main-section").style.display = "none";
        document.getElementById("chat-list-section").style.display = "block";
        try {
            const response = await sendRequestPlaybook(chatInput2.value);
            chatInput2.value = '';
            removeTemporaryBlock();
            addChatBlock(response, "bot", "continuous"); 
        } catch (error) {
            console.error(error);
            if (error.message.includes('500')) {
                window.location.href = '/error';
            }
        }
        chatInput2.disabled = false;
    }
}

function removeTemporaryBlock() {
    var tempBlock = document.getElementById('temporary-block');
    if (tempBlock) {
        tempBlock.parentNode.removeChild(tempBlock);
    }
}

function checkInputContinuousPlaybook() {
    const inputField2 = document.getElementById('chat-input-2');
    const submitButton2 = document.getElementById('button2');
    submitButton2.disabled = inputField2.value.trim().length < 2;
}

document.addEventListener('keydown', async function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent the default Enter key action

        const chatInput1 = document.getElementById('chat-input-1');
        const chatInput2 = document.getElementById('chat-input-2');

        // Determine which function to call based on which input is focused
        if (document.activeElement === chatInput1) {
            await handleSubmitInitialPlaybook();
        } else if (document.activeElement === chatInput2) {
            await handleSubmitContinuousPlaybook();
        }
    }
});