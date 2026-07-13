const form = document.querySelector("#predict-form");
const modelGrid = document.querySelector("#model-grid");
const imageInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const uploadTitle = document.querySelector("#upload-title");
const uploadSubtitle = document.querySelector("#upload-subtitle");
const preview = document.querySelector("#preview");
const submitButton = document.querySelector("#submit-button");
const resultState = document.querySelector("#result-state");
const scoresCard = document.querySelector("#scores-card");
const scoresList = document.querySelector("#scores-list");
const toggleDetailsButton = document.querySelector("#toggle-details");
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:5000" : "";

const fallbackModels = [
  {
    filename: "model_linear1.json",
    model_type: "Lineaire",
    accuracy: 0.8925831202046036,
    class_names: ["FPS", "METROIDVANIA", "MOBA"],
    offline: true,
  },
  {
    filename: "model_rbf2.json",
    model_type: "ovr_rbf",
    accuracy: 0.907928388746803,
    class_names: ["FPS", "METROIDVANIA", "MOBA"],
    offline: true,
  },
  {
    filename: "model_mlp.json",
    model_type: "MLP",
    accuracy: null,
    class_names: ["FPS", "METROIDVANIA", "MOBA"],
    comingSoon: true,
  },
];

const state = {
  selectedModel: null,
  selectedFile: null,
  apiAvailable: false,
  hasSuccessfulPrediction: false,
  isAnalyzing: false,
};

async function loadModels() {
  try {
    const response = await fetch(`${API_BASE}/api/models`);

    if (!response.ok) {
      throw new Error("API indisponible");
    }

    const models = await response.json();
    state.apiAvailable = true;
    renderModels(withModelPlaceholders(models.length ? models : fallbackModels));
  } catch {
    state.apiAvailable = false;
    renderModels(withModelPlaceholders(fallbackModels));
  }
}

function withModelPlaceholders(models) {
  const hasMlp = models.some((model) => isMlpModelType(model.model_type));

  if (hasMlp) {
    return models;
  }

  return [
    ...models,
    {
      filename: "model_mlp.json",
      model_type: "MLP",
      accuracy: null,
      class_names: ["FPS", "METROIDVANIA", "MOBA"],
      comingSoon: true,
    },
  ];
}

function renderModels(models) {
  modelGrid.innerHTML = "";

  for (const model of models) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `model-card${model.comingSoon ? " model-card--disabled" : ""}`;
    card.dataset.filename = model.filename;
    card.disabled = Boolean(model.comingSoon);
    card.innerHTML = `
      <strong>${formatModelName(model)}</strong>
      <div class="model-meta">
        <span class="${model.comingSoon ? "coming-soon" : ""}">
          ${model.comingSoon ? "À venir" : formatAccuracy(model.accuracy)}
        </span>
        <span>${formatModelType(model.model_type)}</span>
      </div>
    `;

    if (!model.comingSoon) {
      card.addEventListener("click", () => selectModel(model.filename));
    }
    modelGrid.appendChild(card);
  }

  const firstAvailableModel = models.find((model) => !model.comingSoon);
  selectModel(firstAvailableModel?.filename ?? null, { autoAnalyze: false });
}

function selectModel(filename, options = {}) {
  const { autoAnalyze = true } = options;
  const previousModel = state.selectedModel;
  state.selectedModel = filename;

  for (const card of modelGrid.querySelectorAll(".model-card")) {
    card.classList.toggle("selected", card.dataset.filename === filename);
  }

  if (
    autoAnalyze &&
    previousModel &&
    previousModel !== filename &&
    state.hasSuccessfulPrediction &&
    state.selectedFile &&
    state.apiAvailable &&
    !state.isAnalyzing
  ) {
    runPrediction();
  }
}

function formatModelName(model) {
  if (model.model_type === "Lineaire") {
    return "Modèle linéaire";
  }

  if (model.model_type === "ovr_rbf") {
    return "Modèle RBF";
  }

  if (isMlpModelType(model.model_type)) {
    return "Modèle MLP";
  }

  return model.filename;
}

function formatModelType(modelType) {
  if (modelType === "Lineaire") {
    return "Linear";
  }

  if (modelType === "ovr_rbf") {
    return "RBF";
  }

  if (isMlpModelType(modelType)) {
    return "MLP";
  }

  return modelType ?? "Modèle";
}

function isMlpModelType(modelType) {
  return ["MLP", "mlp", "PMC", "pmc", "MultilayerPerceptron"].includes(modelType);
}

function formatAccuracy(accuracy) {
  return accuracy == null ? "Accuracy non renseignée" : `${(accuracy * 100).toFixed(1)}% accuracy`;
}

function setSelectedFile(file) {
  state.selectedFile = file;

  if (!file) {
    preview.innerHTML = '<div class="empty-preview"><span>Aucune image sélectionnée</span></div>';
    preview.classList.remove("has-image");
    dropZone.classList.remove("has-file");
    uploadTitle.textContent = "Importer une image";
    uploadSubtitle.textContent = "Glissez une image ici ou cliquez pour choisir un fichier";
    return;
  }

  const image = document.createElement("img");
  image.src = URL.createObjectURL(file);
  image.alt = file.name;

  preview.innerHTML = "";
  preview.classList.add("has-image");
  preview.appendChild(image);
  dropZone.classList.add("has-file");
  uploadTitle.textContent = "Image importée";
  uploadSubtitle.textContent = file.name;
  state.hasSuccessfulPrediction = false;
  scoresCard.hidden = true;
  setIdleResult("Image prête", "Vous pouvez lancer l’analyse.");
}

function setIdleResult(title = "En attente", message = "La prédiction apparaîtra ici.") {
  resultState.className = "result-card idle";
  resultState.innerHTML = `
    <p class="result-kicker">Résultat</p>
    <strong>${title}</strong>
    <span>${message}</span>
  `;
}

function setLoadingResult() {
  resultState.className = "result-card idle";
  resultState.innerHTML = `
    <p class="result-kicker">Analyse</p>
    <strong>En cours...</strong>
    <span>La capture est envoyée au modèle sélectionné.</span>
  `;
}

function setErrorResult(message) {
  resultState.className = "result-card error";
  resultState.innerHTML = `
    <p class="result-kicker">Erreur</p>
    <strong>Impossible</strong>
    <span>${message}</span>
  `;
}

function setSuccessResult(payload) {
  resultState.className = "result-card success";
  resultState.innerHTML = `
    <p class="result-kicker">Genre détecté</p>
    <strong>${formatClassLabel(payload.prediction.label)}</strong>
    <span>Résultat obtenu avec ${formatModelType(payload.model.model_type)}.</span>
  `;

  renderScores(payload.outputs);
}

function formatClassLabel(label) {
  if (label === "METROIDVANIA") {
    return "Metroidvania";
  }

  return label;
}

function renderScores(outputs) {
  const values = outputs.map((output) => output.value);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const scoreRange = maxValue - minValue || 1;

  scoresCard.hidden = false;
  scoresList.hidden = true;
  toggleDetailsButton.textContent = "Voir les scores du modèle";

  scoresList.innerHTML = outputs
    .map((output) => {
      const normalizedScore = (output.value - minValue) / scoreRange;
      const width = Math.max(8, Math.round(normalizedScore * 100));
      return `
        <div class="score-row ${output.winner ? "winner" : ""}">
          <div class="score-header">
            <strong>${formatClassLabel(output.label)}</strong>
            <span>${Number(output.value).toFixed(3)}</span>
          </div>
          <div class="score-bar">
            <span style="--bar-width: ${width}%"></span>
          </div>
        </div>
      `;
    })
    .join("");
}

imageInput.addEventListener("change", () => {
  setSelectedFile(imageInput.files[0] ?? null);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");

  const file = event.dataTransfer.files[0];
  if (!file) {
    return;
  }

  imageInput.files = event.dataTransfer.files;
  setSelectedFile(file);
});

toggleDetailsButton.addEventListener("click", () => {
  const nextHiddenState = !scoresList.hidden;
  scoresList.hidden = nextHiddenState;
  toggleDetailsButton.textContent = nextHiddenState
    ? "Voir les scores du modèle"
    : "Masquer les détails";
});

async function runPrediction() {
  if (!state.selectedModel) {
    setErrorResult("Sélectionnez un modèle avant de lancer l’analyse.");
    return;
  }

  if (!state.selectedFile) {
    setErrorResult("Sélectionnez une image avant de lancer l’analyse.");
    return;
  }

  if (!state.apiAvailable) {
    setErrorResult("Lancez le serveur local pour activer la prédiction.");
    return;
  }

  const formData = new FormData();
  formData.append("model", state.selectedModel);
  formData.append("image", state.selectedFile);

  setLoadingResult();
  state.isAnalyzing = true;
  submitButton.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error ?? "Erreur inconnue.");
    }

    setSuccessResult(payload);
    state.hasSuccessfulPrediction = true;
  } catch (error) {
    setErrorResult(error.message);
  } finally {
    state.isAnalyzing = false;
    submitButton.disabled = false;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runPrediction();
});

loadModels();
