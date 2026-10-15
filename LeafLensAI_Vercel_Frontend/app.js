const input = document.getElementById('imageInput');
const dropzone = document.getElementById('dropzone');
const previewWrap = document.getElementById('previewWrap');
const preview = document.getElementById('preview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const buttonText = document.getElementById('buttonText');
const spinner = document.getElementById('spinner');
const emptyState = document.getElementById('emptyState');
const result = document.getElementById('result');
const notice = document.getElementById('notice');
let selectedFile = null;

function showNotice(message, isError = false) {
  notice.textContent = message;
  notice.classList.remove('hidden', 'error');
  if (isError) notice.classList.add('error');
}
function clearNotice() { notice.classList.add('hidden'); notice.classList.remove('error'); notice.textContent = ''; }
function setFile(file) {
  clearNotice();
  if (!file) return;
  if (!['image/jpeg', 'image/png'].includes(file.type)) return showNotice('Please select a JPG, JPEG, or PNG image.', true);
  if (file.size > 4 * 1024 * 1024) return showNotice('Image is larger than 4 MB. Please choose a smaller image.', true);
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
  previewWrap.classList.remove('hidden');
  dropzone.classList.add('hidden');
  analyzeBtn.disabled = false;
  emptyState.classList.remove('hidden');
  result.classList.add('hidden');
}
function resetFile() {
  if (preview.src.startsWith('blob:')) URL.revokeObjectURL(preview.src);
  selectedFile = null; input.value = ''; preview.src = '';
  previewWrap.classList.add('hidden'); dropzone.classList.remove('hidden');
  analyzeBtn.disabled = true; result.classList.add('hidden'); emptyState.classList.remove('hidden'); clearNotice();
}
input.addEventListener('change', () => setFile(input.files[0]));
removeBtn.addEventListener('click', resetFile);
['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, e => { e.preventDefault(); dropzone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, e => { e.preventDefault(); dropzone.classList.remove('dragging'); }));
dropzone.addEventListener('drop', e => setFile(e.dataTransfer.files[0]));

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  clearNotice(); analyzeBtn.disabled = true; buttonText.textContent = 'Analyzing…'; spinner.classList.remove('hidden');
  const form = new FormData(); form.append('image', selectedFile);
  try {
    const response = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Prediction request failed (${response.status}).`);
    if (!data.label || typeof data.confidence !== 'number') throw new Error('The prediction service returned an unexpected response.');
    const label = String(data.label).replaceAll('_', ' ');
    document.getElementById('predictionLabel').textContent = label.replace(/\b\w/g, c => c.toUpperCase());
    const confidence = Math.max(0, Math.min(100, data.confidence));
    document.getElementById('confidenceText').textContent = `${confidence.toFixed(1)}%`;
    document.getElementById('confidenceBar').style.width = `${confidence}%`;
    const lower = label.toLowerCase();
    let title = 'Possible issue detected';
    let message = 'This is an AI screening result, not a confirmed diagnosis. Check the plant and consult a local agricultural expert before treatment.';
    if (lower.includes('healthy')) { title = 'Healthy-looking leaf'; message = 'The model classified this image as healthy. Continue regular monitoring and good plant care.'; }
    if (confidence < 60) message += ' The model confidence is low; try another clear photo or verify manually.';
    document.getElementById('interpretation').innerHTML = `<strong>${title}</strong><br>${message}`;
    emptyState.classList.add('hidden'); result.classList.remove('hidden');
  } catch (error) {
    showNotice(`${error.message} Check that the AI backend is deployed and the LEAFLENS_API_URL environment variable is configured.`, true);
  } finally {
    analyzeBtn.disabled = !selectedFile; buttonText.textContent = 'Analyze leaf'; spinner.classList.add('hidden');
  }
});
