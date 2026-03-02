document.addEventListener("DOMContentLoaded", function() {

    // Interval button functionality
    const intervalButton = document.getElementById("intervalbtn");
    intervalButton.addEventListener('click', function(e) {
        alert("WIP");
    });

    // Search button functionality
    const searchButton = document.querySelector('.topbarbtn img[alt="Search Symbol"]').parentElement;
    const searchsymbol = document.getElementById("searchsymbol");
    const closebtn = document.getElementsByClassName("close")[0];

    searchButton.addEventListener('click', function(event) {
        event.preventDefault();
        searchsymbol.style.display = "block";
    });

    closebtn.onclick = function() {
        searchsymbol.style.display = "none";
    }

    window.onclick = function(event) {
        if (event.target == searchsymbol) {
            searchsymbol.style.display = "none";
        }
    }

    // Search symbol functionality
    const searchInput = document.getElementById("searchinput");
    const suggestions = document.getElementById("suggestions");

    searchInput.addEventListener("input", function() {
        const filter = searchInput.value.toLowerCase();
        const options = suggestions.getElementsByTagName("option");
        for (let i = 0; i < options.length; i++) {
            const txtValue = options[i].value;
            if (txtValue.toLowerCase().indexOf(filter) > -1) {
                options[i].style.display = "";
            } else {
                options[i].style.display = "none";
            }
        }
    });

    // Pressing "Enter" when searching for symbol
    searchInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            event.preventDefault();
            const symbol = searchInput.value.trim().toLowerCase();
            if (symbol) {
                // Check if the button already exists
                const existingButton = Array.from(document.querySelectorAll('.crypto-btn')).find(btn => btn.textContent.trim().toLowerCase().startsWith(symbol));
                if (existingButton) {
                    alert(`${symbol} is already in the sidebar.`);
                    return;
                }
                
                // Send AJAX request to update sidebar
                fetch('/update_sidebar', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify({ symbol: symbol })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const sidebar = document.getElementById("sidebar");
                        sidebar.innerHTML = '';
                        const watchlistitem = document.createElement("button");
                        watchlistitem.className = "crypto-btn";
                        watchlistitem.innerHTML = `${symbol} <span class="delete-btn">X</span>`;
                        sidebar.appendChild(watchlistitem);

                        // Add delete functionality to button
                        watchlistitem.querySelector('.delete-btn').addEventListener('click', function(event) {
                            event.stopPropagation();
                            watchlistitem.remove();
                        });

                        closeSearchPopup();
                    }
                });

                // Check for specific symbols and add respective buttons
                if (symbol === 'bitcoin' || symbol === 'ethereum') {
                    const sidebar = document.getElementById("sidebar");
                    const watchlistitem = document.createElement("button");
                    watchlistitem.className = "crypto-btn";
                    watchlistitem.innerHTML = `${symbol} <span class="delete-btn">X</span>`;
                    sidebar.appendChild(watchlistitem);

                    // Add delete functionality to button
                    watchlistitem.querySelector('.delete-btn').addEventListener('click', function(event) {
                        event.stopPropagation();
                        watchlistitem.remove();
                    });

                    closeSearchPopup();
                }
            }
        }
    });

    // Function to get CSRF token
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Delete button functionality
    document.querySelectorAll('.delete-btn').forEach(function(deleteBtn) {
        deleteBtn.addEventListener('click', function(event) {
            event.stopPropagation();
            const parentButton = deleteBtn.parentElement;
            parentButton.remove();
        });
    });

});

function closeSearchPopup() {
    const searchPopup = document.getElementById("searchsymbol");
    if (searchPopup) {
        searchPopup.style.display = "none";
    }
}