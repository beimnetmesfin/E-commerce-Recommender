
"use strict";

// Ye-buna storefront prototype.
// Events are stored in this browser, not yet sent to FastAPI.

const $ = (selector) => document.querySelector(selector);

const state = {
  products: [],
  filteredProducts: [],
  page: 1,
  pageSize: 20,
  userId: "user_001",
  cart: {},
  liked: new Set(),
};

const STORAGE_PREFIX = "yebuna_demo_v1";
const $grid = $("#productGrid");
const $recommendationGrid = $("#recommendationGrid");

function storageKey(type, userId = state.userId) {
  return `${STORAGE_PREFIX}_${type}_${userId}`;
}

function readStorage(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    showToast("Browser storage is unavailable.");
  }
}

// Parse CSV correctly when values contain commas or quoted text.
function parseCSV(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && quoted && next === '"') {
      cell += '"';
      i++;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i++;
      row.push(cell);
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  if (cell !== "" || row.length) {
    row.push(cell);
    if (row.some((value) => value.trim() !== "")) rows.push(row);
  }

  if (!rows.length) return [];

  const headers = rows[0].map((header) => header.trim().replace(/^\uFEFF/, ""));

  return rows.slice(1).map((values) => {
    const product = {};
    headers.forEach((header, index) => {
      product[header] = (values[index] ?? "").trim();
    });
    return product;
  });
}

function numberValue(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function productId(product) {
  return String(product.item_id ?? "");
}

function formatPrice(product) {
  const price = numberValue(product.price);
  const currency = String(product.currency || "ETB").toUpperCase();
  return `${currency} ${price.toLocaleString(undefined, {
    maximumFractionDigits: 2,
  })}`;
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function safeImageURL(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";

  try {
    const url = new URL(raw, window.location.href);
    if (url.protocol === "https:" || url.protocol === "http:") {
      return url.href;
    }
  } catch {
    // Invalid image URL: use the placeholder below.
  }

  return "";
}

function imageMarkup(product, extraClass = "") {
  const url = safeImageURL(product.image_url);
  const title = escapeHTML(product.title || "Product image");

  if (!url) {
    return `<div class="product-image ${extraClass}" role="img"
      aria-label="${title}" style="display:grid;place-items:center">
      <span>No image available</span>
    </div>`;
  }

  return `<img class="product-image ${extraClass}"
    src="${escapeHTML(url)}"
    alt="${title}"
    loading="lazy"
    referrerpolicy="no-referrer"
    onerror="this.onerror=null;this.replaceWith(Object.assign(document.createElement('div'),{className:'product-image',textContent:'Image unavailable',style:'display:grid;place-items:center'}))">`;
}

function getEvents() {
  return readStorage(storageKey("events"), []);
}


async function recordEvent(product, eventType) {
  const event = {
    user_id: state.userId,
    item_id: String(productId(product)),
    event_type: eventType,
    timestamp: new Date().toISOString(),
  };

  // 1. Save the event in the browser
  const events = getEvents();
  events.push(event);
  writeStorage(storageKey("events"), events);

  // 2. Update the website interface
  updateActivitySummary();

  // 3. Send the event to FastAPI
  try {
    const response = await fetch("http://127.0.0.1:8000/events", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: event.user_id,
        item_id: event.item_id,
        event_type: event.event_type,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }

    const result = await response.json();
    console.log("Event saved by FastAPI:", result);

  } catch (error) {
    console.error("Failed to send event to FastAPI:", error);
  }

  // 4. Refresh ML recommendations
  await updateRecommendations();
}



function getLikes() {
  return readStorage(storageKey("likes"), []);
}

function saveLikes(likes) {
  writeStorage(storageKey("likes"), likes);
  state.liked = new Set(likes.map(String));
  updateActivitySummary();
}

function loadUserState() {
  state.cart = readStorage(storageKey("cart"), {});
  state.liked = new Set(getLikes().map(String));
  updateCartCount();
  updateActivitySummary();
}


async function loadProducts() {
  const status = $("#statusMessage");
  status.textContent = "Loading products from data/products.csv...";
  status.classList.remove("error");

  try {
    const response = await fetch("./data/products.csv", {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(
        `Could not load data/products.csv (HTTP ${response.status}).`
      );
    }

    const csv = await response.text();
    const products = parseCSV(csv);

    if (!products.length || !("item_id" in products[0])) {
      throw new Error(
        "The CSV is empty or does not contain an item_id column."
      );
    }

    state.products = products.filter(
      (product) => productId(product) !== ""
    );

    state.products.forEach((product) => {
      product._id = productId(product);
    });

    $("#heroProductCount").textContent =
      state.products.length.toLocaleString();

    populateCategories();
    applyFilters();

    status.textContent =
      `Successfully loaded ${state.products.length.toLocaleString()} products from your CSV.`;

    // Load recommendations after products are ready
    await updateRecommendations();

  } catch (error) {
    console.error(error);
    status.classList.add("error");

    status.textContent =
      `${error.message} Copy your products.csv into the website's data folder, ` +
      "then run the website using a local server. Do not open index.html directly.";

    $("#resultCount").textContent = "Catalog unavailable";
  }
}


function populateCategories() {
  const select = $("#categoryFilter");
  const categories = [...new Set(
    state.products
      .map((product) => product.category)
      .filter(Boolean)
  )].sort((a, b) => a.localeCompare(b));

  select.innerHTML =
    '<option value="">All categories</option>' +
    categories.map((category) =>
      `<option value="${escapeHTML(category)}">${escapeHTML(category)}</option>`
    ).join("");
}

function applyFilters() {
  const query = $("#searchInput").value.trim().toLowerCase();
  const category = $("#categoryFilter").value;

  state.filteredProducts = state.products.filter((product) => {
    const searchable = [
      product.title,
      product.description,
      product.category,
      product.subcategory,
      product.tags,
      product.creator_name,
    ].join(" ").toLowerCase();

    return (!query || searchable.includes(query)) &&
      (!category || product.category === category);
  });

  const sort = $("#sortFilter").value;

  if (sort === "price-low") {
    state.filteredProducts.sort(
      (a, b) => numberValue(a.price) - numberValue(b.price)
    );
  } else if (sort === "price-high") {
    state.filteredProducts.sort(
      (a, b) => numberValue(b.price) - numberValue(a.price)
    );
  } else if (sort === "rating") {
    state.filteredProducts.sort(
      (a, b) => numberValue(b.average_rating) - numberValue(a.average_rating)
    );
  } else if (sort === "popular") {
    state.filteredProducts.sort(
      (a, b) => numberValue(b.total_sales) - numberValue(a.total_sales)
    );
  }

  state.page = 1;
  renderProducts();
}

function productCard(product) {
  const id = productId(product);
  const liked = state.liked.has(id);
  const stock = numberValue(product.stock);
  const rating = numberValue(product.average_rating);
  const reviews = numberValue(product.total_reviews);

  return `
    <article class="product-card">
      <div class="product-image-wrap">
        ${imageMarkup(product)}
        <span class="category-pill">${escapeHTML(product.category || "General")}</span>
        <button class="like-button ${liked ? "liked" : ""}"
          data-action="like" data-id="${escapeHTML(id)}"
          aria-label="${liked ? "Unlike" : "Like"} ${escapeHTML(product.title)}"
          aria-pressed="${liked}">${liked ? "♥" : "♡"}</button>
      </div>
      <div class="product-info">
        <h3 class="product-title">${escapeHTML(product.title || "Untitled product")}</h3>
        <div class="product-seller">By ${escapeHTML(product.creator_name || "Ye-buna seller")}</div>
        <div class="product-rating">★ ${rating.toFixed(1)} <span class="muted">(${reviews} reviews)</span></div>
        <div class="product-bottom">
          <span class="product-price">${escapeHTML(formatPrice(product))}</span>
          <button class="add-button" data-action="add" data-id="${escapeHTML(id)}"
            aria-label="Add ${escapeHTML(product.title)} to cart"
            ${stock <= 0 ? "disabled" : ""}>+</button>
        </div>
        <div class="stock-note">${stock > 0 ? `${stock} in stock` : "Out of stock"}</div>
        <button type="button" class="secondary-button full-button" style="margin-top:10px"
          data-action="details" data-id="${escapeHTML(id)}">View details</button>
      </div>
    </article>`;
}

function renderProducts() {
  const total = state.filteredProducts.length;
  const pageCount = Math.max(1, Math.ceil(total / state.pageSize));

  if (state.page > pageCount) state.page = pageCount;

  const start = (state.page - 1) * state.pageSize;
  const pageProducts = state.filteredProducts.slice(start, start + state.pageSize);

  $grid.innerHTML = pageProducts.map(productCard).join("");

  $("#resultCount").textContent =
    `${total.toLocaleString()} product${total === 1 ? "" : "s"} found`;

  $("#pageInfo").textContent = `Page ${state.page} of ${pageCount}`;
  $("#previousPage").disabled = state.page <= 1;
  $("#nextPage").disabled = state.page >= pageCount;

  if (!pageProducts.length) {
    $grid.innerHTML = `<p class="muted">No products match your search. Try another keyword or category.</p>`;
  }
}

function findProduct(id) {
  return state.products.find((product) => productId(product) === String(id));
}


function openProductDetails(product) {
  if (!product) return;

  const id = productId(product);

  // Render the product details first.
  $("#productDetails").innerHTML = `
    <div class="modal-product-layout">
      ${imageMarkup(product)}
      <div>
        <p class="eyebrow">${escapeHTML(product.category || "PRODUCT")}</p>
        <h2 id="modalTitle">${escapeHTML(product.title || "Product details")}</h2>
        <p class="muted">Sold by ${escapeHTML(product.creator_name || "Ye-buna seller")}</p>
        <div class="product-rating">
          ★ ${numberValue(product.average_rating).toFixed(1)}
          (${numberValue(product.total_reviews)} reviews)
        </div>
        <p class="modal-price">${escapeHTML(formatPrice(product))}</p>
        <p class="modal-description">${escapeHTML(product.description || "No description provided.")}</p>
        <p class="muted small-text">Product ID: ${escapeHTML(id)}</p>
        <p class="muted small-text">Stock: ${numberValue(product.stock)}</p>
        <div class="modal-actions">
          <button type="button" class="primary-button"
            data-action="add" data-id="${escapeHTML(id)}">
            Add to cart
          </button>
          <button type="button" class="secondary-button"
            data-action="like" data-id="${escapeHTML(id)}">
            ${state.liked.has(id) ? "♥ Liked" : "♡ Like"}
          </button>
          <button type="button" class="secondary-button"
            data-action="skip" data-id="${escapeHTML(id)}">
            Not interested
          </button>
        </div>
      </div>
    </div>`;

  // Open the modal immediately.
  $("#productModal").hidden = false;

  // Record events in the background; do not wait for the API.
  void recordEvent(product, "view");
  void recordEvent(product, "click");
}


function closeProductDetails() {
  console.trace("PRODUCT MODAL CLOSED");

  const modal = document.querySelector("#productModal");

  console.log("Modal before closing:", {
    hidden: modal.hidden,
    time: new Date().toISOString()
  });

  modal.hidden = true;
}

function toggleLike(product) {
  if (!product) return;

  const id = productId(product);
  const likes = getLikes();
  const exists = likes.includes(id);

  if (exists) {
    saveLikes(likes.filter((likedId) => likedId !== id));
    showToast("Removed from your liked products.");
  } else {
    saveLikes([...likes, id]);
    recordEvent(product, "like");
    showToast("Added to your liked products.");
  }

  renderProducts();
  updateRecommendations();

  if (!$("#productModal").hidden) openProductDetailsWithoutView(product);
}

function openProductDetailsWithoutView(product) {
  // Refresh the like button without recording another view.
  const id = productId(product);
  const likeButton = $("#productDetails").querySelector('[data-action="like"]');
  if (likeButton) {
    likeButton.textContent = state.liked.has(id) ? "♥ Liked" : "♡ Like";
  }
}

function addToCart(product) {
  if (!product) return;

  const id = productId(product);
  const stock = numberValue(product.stock);
  const current = numberValue(state.cart[id]);

  if (stock <= current) {
    showToast("No more stock is available for this demo product.");
    return;
  }

  state.cart[id] = current + 1;
  writeStorage(storageKey("cart"), state.cart);
  recordEvent(product, "add_to_cart");
  updateCartCount();
  showToast("Product added to your cart.");
}

function updateCartCount() {
  const count = Object.values(state.cart).reduce(
    (total, quantity) => total + numberValue(quantity), 0
  );
  $("#cartCount").textContent = count;
}

function openCart() {
  const entries = Object.entries(state.cart)
    .filter(([, quantity]) => numberValue(quantity) > 0);

  let total = 0;

  $("#cartItems").innerHTML = entries.map(([id, quantity]) => {
    const product = findProduct(id);
    if (!product) return "";

    total += numberValue(product.price) * numberValue(quantity);

    return `
      <div class="cart-row">
        <div>
          <strong>${escapeHTML(product.title)}</strong>
          <small>${escapeHTML(formatPrice(product))} × ${numberValue(quantity)}</small>
        </div>
        <div>
          <strong>${escapeHTML((numberValue(product.price) * numberValue(quantity)).toLocaleString())}
          ${escapeHTML(String(product.currency || "ETB").toUpperCase())}</strong>
          <button data-action="remove" data-id="${escapeHTML(id)}">Remove</button>
        </div>
      </div>`;
  }).join("");

  if (!entries.length) {
    $("#cartItems").innerHTML = `<p class="muted">Your cart is empty. Explore the marketplace to find something you like.</p>`;
  }

  $("#cartTotal").textContent = total.toLocaleString(undefined, {
    style: "currency",
    currency: entries.length
      ? (findProduct(entries[0][0])?.currency || "ETB")
      : "ETB",
    maximumFractionDigits: 2,
  });

  $("#checkoutButton").disabled = !entries.length;
  $("#cartModal").hidden = false;
}

function removeFromCart(id) {
  delete state.cart[id];
  writeStorage(storageKey("cart"), state.cart);
  updateCartCount();
  openCart();
}

function simulateCheckout() {
  const entries = Object.entries(state.cart)
    .filter(([, quantity]) => numberValue(quantity) > 0);

  if (!entries.length) return;

  for (const [id, quantity] of entries) {
    const product = findProduct(id);
    if (!product) continue;

    for (let i = 0; i < numberValue(quantity); i++) {
      recordEvent(product, "purchase");
    }
  }

  state.cart = {};
  writeStorage(storageKey("cart"), state.cart);
  updateCartCount();
  $("#cartModal").hidden = true;
  showToast("Demo checkout completed. No payment was processed.");
}

function updateActivitySummary() {
  $("#eventCount").textContent = getEvents().length;
  $("#likedCount").textContent = getLikes().length;
}


async function updateRecommendations() {
  const grid = $("#recommendationGrid");
  const note = $("#recommendationNote");

  if (!state.products.length) {
    return;
  }

  grid.innerHTML = '<p class="muted">Loading recommendations...</p>';

  try {
    const response = await fetch(
      `http://127.0.0.1:8000/recommendations/${encodeURIComponent(state.userId)}?top_k=4`
    );

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    const result = await response.json();
    const recommendations = result.recommendations || [];

    if (recommendations.length === 0) {
      grid.innerHTML =
        '<p class="muted">No recommendations available yet.</p>';

      note.textContent =
        "Explore products to build your recommendations.";
      return;
    }

    grid.innerHTML = recommendations
      .map((item) => {
        const catalogProduct = findProduct(item.item_id);

        const product = catalogProduct
          ? { ...catalogProduct, ...item }
          : item;

        return productCard(product);
      })
      .join("");

    note.textContent =
      `ML recommendations for ${state.userId}.`;
  } catch (error) {
    console.error("Recommendation API error:", error);

    grid.innerHTML =
      '<p class="muted">Could not load recommendations. Check that the API is running.</p>';

    note.textContent =
      "Recommendation service unavailable.";
  }
}


function exportEventsCSV() {
  const events = getEvents();

  if (!events.length) {
    showToast("There are no events to export yet.");
    return;
  }

  const headers = ["user_id", "item_id", "event_type", "timestamp"];
  const lines = [
    headers.join(","),
    ...events.map((event) =>
      headers.map((header) => {
        const value = String(event[header] ?? "");
        return `"${value.replace(/"/g, '""')}"`;
      }).join(",")
    ),
  ];

  const blob = new Blob([lines.join("\r\n")], {
    type: "text/csv;charset=utf-8;",
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.userId}_events.csv`;
  link.click();
  URL.revokeObjectURL(url);
  showToast("Event CSV exported.");
}

let toastTimeout;

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");

  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.remove("show"), 2600);
}

// Delegate product actions so dynamically rendered cards work too.
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (!button) return;

  const product = findProduct(button.dataset.id);
  if (!product) return;

  switch (button.dataset.action) {
    case "details":
      openProductDetails(product);
      break;
    case "like":
      toggleLike(product);
      break;
    case "add":
      addToCart(product);
      break;
    case "skip":
      recordEvent(product, "skip");
      closeProductDetails();
      showToast("Feedback recorded for this demo user.");
      break;
    case "remove":
      removeFromCart(productId(product));
      break;
  }
});

$("#searchInput").addEventListener("input", applyFilters);
$("#categoryFilter").addEventListener("change", applyFilters);
$("#sortFilter").addEventListener("change", applyFilters);

$("#previousPage").addEventListener("click", () => {
  if (state.page > 1) {
    state.page--;
    renderProducts();
    $("#products").scrollIntoView({ behavior: "smooth" });
  }
});

$("#nextPage").addEventListener("click", () => {
  const pages = Math.ceil(state.filteredProducts.length / state.pageSize);
  if (state.page < pages) {
    state.page++;
    renderProducts();
    $("#products").scrollIntoView({ behavior: "smooth" });
  }
});

$("#demoUser").addEventListener("change", (event) => {
  state.userId = event.target.value;
  loadUserState();
  renderProducts();
  updateRecommendations();
  showToast(`Switched to ${state.userId}.`);
});

$("#cartButton").addEventListener("click", openCart);
$("#closeProductModal").addEventListener("click", closeProductDetails);
$("#closeCartModal").addEventListener("click", () => {
  $("#cartModal").hidden = true;
});
$("#checkoutButton").addEventListener("click", simulateCheckout);
$("#exportEvents").addEventListener("click", exportEventsCSV);

$("#productModal").addEventListener("click", (event) => {
  if (event.target === $("#productModal")) closeProductDetails();
});
$("#cartModal").addEventListener("click", (event) => {
  if (event.target === $("#cartModal")) $("#cartModal").hidden = true;
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeProductDetails();
    $("#cartModal").hidden = true;
  }
});

loadUserState();
loadProducts();
