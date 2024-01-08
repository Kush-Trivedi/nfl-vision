function showSpinner(event) {
    // Prevent the default form submission
    event.preventDefault();

    // Get form and input values
    var form = document.getElementById("myForm");
    var gameId = document.querySelector('input[name="game_id"]').value;
    var playId = document.querySelector('input[name="play_id"]').value;

    // Check if input fields are empty
    if (!gameId || !playId) {
        // If either field is empty, do not proceed with submission
        alert("Both Game ID and Play ID are required.");
        return;
    }

    // Initialize components
    var spinner = document.querySelector(".spinner-border");
    var button = document.getElementById("submitBtn");
    var paperPlaneIcon = document.getElementById("paperPlaneIcon");

    // Show spinner and disable button
    spinner.style.display = "block";
    paperPlaneIcon.style.display = "none";
    button.disabled = true;

    // Submit the form
    form.submit();

    // Disable form elements
    for (var i = 0, len = form.elements.length; i < len; ++i) {
        form.elements[i].disabled = true;
    }
}
