document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("inference-form");
    const statusMessage = document.getElementById("status-message");
    const runButton = document.getElementById("run-btn");
    const mediaInput = document.getElementById("media-input");
    const selectedFile = document.getElementById("selected-file");
    const previewWrapper = document.getElementById("preview-wrapper");
    const mediaPreview = document.getElementById("media-preview");
    const thresholdRange = document.getElementById("threshold-range");
    const thresholdValue = document.getElementById("threshold-value");
    const resultsPanel = document.getElementById("results-panel");
    const riskBanner = document.getElementById("risk-banner");
    const resultCards = document.getElementById("result-cards");
    const downloadResultsButton = document.getElementById("download-results-btn");

    const emergencyPanel = document.getElementById("emergency-panel");
    const emergencyStatus = document.getElementById("emergency-status");
    const prepareEmergencyButton = document.getElementById("prepare-emergency-btn");
    const callEmergencyLink = document.getElementById("call-emergency-link");
    const copySummaryButton = document.getElementById("copy-summary-btn");
    const incidentSummary = document.getElementById("incident-summary");
    const incidentLocation = document.getElementById("incident-location");
    const incidentNotes = document.getElementById("incident-notes");

    const manualLocation = document.getElementById("manual-location");
    const manualIncidentType = document.getElementById("manual-incident-type");
    const manualSeverity = document.getElementById("manual-severity");
    const manualNotes = document.getElementById("manual-notes");
    const manualEmergencyButton = document.getElementById("manual-emergency-btn");
    const manualCallLink = document.getElementById("manual-call-link");
    const manualCopySummaryButton = document.getElementById("manual-copy-summary-btn");
    const manualEmergencyStatus = document.getElementById("manual-emergency-status");
    const manualIncidentSummary = document.getElementById("manual-incident-summary");

    const faqSearch = document.getElementById("faq-search");
    const faqList = document.getElementById("faq-list");

    const contactForm = document.getElementById("contact-form");
    const contactName = document.getElementById("contact-name");
    const contactEmail = document.getElementById("contact-email");
    const contactTopic = document.getElementById("contact-topic");
    const contactMessage = document.getElementById("contact-message");
    const contactSubmitButton = document.getElementById("contact-submit-btn");
    const contactClearButton = document.getElementById("contact-clear-btn");
    const contactStatus = document.getElementById("contact-status");

    const routePlanButton = document.getElementById("route-plan-btn");
    const routeCopyButton = document.getElementById("route-copy-btn");
    const routeStatus = document.getElementById("route-status");
    const routeSummary = document.getElementById("route-summary");
    const routeGoalCount = document.getElementById("route-goal-count");

    if (
        !form ||
        !statusMessage ||
        !runButton ||
        !mediaInput ||
        !selectedFile ||
        !previewWrapper ||
        !mediaPreview ||
        !thresholdRange ||
        !thresholdValue ||
        !resultsPanel ||
        !riskBanner ||
        !resultCards ||
        !downloadResultsButton ||
        !emergencyPanel ||
        !emergencyStatus ||
        !prepareEmergencyButton ||
        !callEmergencyLink ||
        !copySummaryButton ||
        !incidentSummary ||
        !incidentLocation ||
        !incidentNotes ||
        !manualLocation ||
        !manualIncidentType ||
        !manualSeverity ||
        !manualNotes ||
        !manualEmergencyButton ||
        !manualCallLink ||
        !manualCopySummaryButton ||
        !manualEmergencyStatus ||
        !manualIncidentSummary ||
        !faqSearch ||
        !faqList ||
        !contactForm ||
        !contactName ||
        !contactEmail ||
        !contactTopic ||
        !contactMessage ||
        !contactSubmitButton ||
        !contactClearButton ||
        !contactStatus
    ) {
        return;
    }

    const inferEndpoint = form.dataset.endpoint;
    const emergencyEndpoint = form.dataset.emergencyEndpoint;
    const emergencyStandaloneEndpoint = form.dataset.emergencyStandaloneEndpoint;
    const contactEndpoint = form.dataset.contactEndpoint;
    const routeEndpoint = form.dataset.routeEndpoint;
    const csrfInput = form.querySelector("input[name='csrfmiddlewaretoken']");

    let latestResults = [];
    let latestResultsJson = "";
    let latestRouteJson = "";
    let previewUrl = null;

    const setStatus = (element, type, text) => {
        element.className = `status ${type || ""}`.trim();
        element.textContent = text;
    };

    const clearResults = () => {
        latestResults = [];
        latestResultsJson = "";
        resultsPanel.hidden = true;
        emergencyPanel.hidden = true;
        resultCards.innerHTML = "";
        riskBanner.textContent = "";
        incidentSummary.hidden = true;
        incidentSummary.textContent = "";
        callEmergencyLink.hidden = true;
        copySummaryButton.hidden = true;
    };

    const updateThresholdDisplay = () => {
        thresholdValue.textContent = Number(thresholdRange.value).toFixed(2);
        if (latestResults.length > 0) {
            renderRiskBanner(latestResults);
            renderResultCards(latestResults);
        }
    };

    const renderRiskBanner = (results) => {
        if (!results || results.length === 0) {
            riskBanner.className = "risk-banner low";
            riskBanner.textContent = "No detections were returned.";
            return;
        }
        const threshold = Number(thresholdRange.value);
        const maxConfidence = Math.max(...results.map((row) => Number(row.confidence) || 0));
        const isHighRisk = maxConfidence >= threshold;
        riskBanner.className = isHighRisk ? "risk-banner high" : "risk-banner low";
        riskBanner.textContent = isHighRisk
            ? `Alert: max confidence ${maxConfidence.toFixed(6)} exceeds threshold ${threshold.toFixed(2)}.`
            : `Nominal: max confidence ${maxConfidence.toFixed(6)} is below threshold ${threshold.toFixed(2)}.`;
    };

    const renderResultCards = (results) => {
        const threshold = Number(thresholdRange.value);
        resultCards.innerHTML = "";
        if (!results || results.length === 0) {
            const empty = document.createElement("p");
            empty.className = "subtext compact";
            empty.textContent = "No results to display.";
            resultCards.appendChild(empty);
            return;
        }

        for (const row of results) {
            const confidence = Number(row.confidence) || 0;
            const high = confidence >= threshold;
            const status = high ? "Above threshold" : "Below threshold";
            const card = document.createElement("article");
            card.className = "result-card";
            card.innerHTML = `
                <div class="result-head-row">
                    <h4>${row.detected_anomaly}</h4>
                    <span class="badge ${high ? "badge-danger" : "badge-safe"}">${status}</span>
                </div>
                <p class="result-model">Model: ${row.model_used}</p>
                <p class="result-confidence">Confidence: ${confidence.toFixed(6)}</p>
                <div class="meter">
                    <div class="meter-fill ${high ? "meter-fill-danger" : "meter-fill-safe"}" style="width: ${Math.min(confidence * 100, 100)}%;"></div>
                </div>
            `;
            resultCards.appendChild(card);
        }
    };

    const resetInferenceEmergencyUi = () => {
        setStatus(emergencyStatus, "", "");
        callEmergencyLink.hidden = true;
        copySummaryButton.hidden = true;
        incidentSummary.hidden = true;
        incidentSummary.textContent = "";
    };

    const updatePreview = () => {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }

        mediaPreview.innerHTML = "";
        const file = mediaInput.files && mediaInput.files[0] ? mediaInput.files[0] : null;
        if (!file) {
            selectedFile.textContent = "No file selected.";
            previewWrapper.hidden = true;
            return;
        }

        selectedFile.textContent = `Selected: ${file.name}`;
        previewUrl = URL.createObjectURL(file);
        previewWrapper.hidden = false;

        if (file.type.startsWith("image/")) {
            const img = document.createElement("img");
            img.src = previewUrl;
            img.alt = "Preview";
            img.className = "media-preview-visual";
            mediaPreview.appendChild(img);
            return;
        }

        if (file.type.startsWith("video/")) {
            const video = document.createElement("video");
            video.src = previewUrl;
            video.controls = true;
            video.className = "media-preview-visual";
            mediaPreview.appendChild(video);
            return;
        }

        mediaPreview.textContent = "Preview unavailable for this file type.";
    };

    const sendJson = async (url, payload) => {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfInput ? csrfInput.value : "",
            },
            credentials: "same-origin",
            body: JSON.stringify(payload),
        });
        const raw = await response.text();
        let data = {};
        try {
            data = raw ? JSON.parse(raw) : {};
        } catch (error) {
            data = { success: false, error: raw || "Non-JSON response from server." };
        }
        if (!response.ok || !data.success) {
            throw new Error(data.error || `Request failed with HTTP ${response.status}.`);
        }
        return data;
    };

    if (
        routePlanButton &&
        routeCopyButton &&
        routeStatus &&
        routeSummary &&
        routeEndpoint
    ) {
        routePlanButton.addEventListener("click", async () => {
            routePlanButton.disabled = true;
            latestRouteJson = "";
            routeCopyButton.hidden = true;
            routeSummary.hidden = true;
            routeSummary.textContent = "";

            const goalCountRaw = routeGoalCount ? routeGoalCount.value.trim() : "";
            if (goalCountRaw) {
                const parsedGoalCount = Number(goalCountRaw);
                if (!Number.isInteger(parsedGoalCount) || parsedGoalCount <= 0) {
                    setStatus(routeStatus, "error", "Goal count must be a positive integer.");
                    routePlanButton.disabled = false;
                    return;
                }
            }

            setStatus(
                routeStatus,
                "",
                "Opening matplotlib selector. Click 1 start + goals on the map, then wait for route inference.",
            );

            try {
                const payload = {};
                if (goalCountRaw) {
                    payload.goal_count = Number(goalCountRaw);
                }
                const data = await sendJson(routeEndpoint, payload);

                latestRouteJson = JSON.stringify(data.route, null, 2);
                routeSummary.textContent = latestRouteJson;
                routeSummary.hidden = false;
                routeCopyButton.hidden = false;

                const completed = data.route && data.route.completed_all_goals ? "yes" : "no";
                const travelTimeSec = Number(data.route.estimated_travel_time_sec || 0).toFixed(2);
                setStatus(
                    routeStatus,
                    "success",
                    `Route generated. Completed all goals: ${completed}. Estimated travel time: ${travelTimeSec}s.`,
                );
            } catch (error) {
                setStatus(routeStatus, "error", error.message);
            } finally {
                routePlanButton.disabled = false;
            }
        });

        routeCopyButton.addEventListener("click", async () => {
            if (!latestRouteJson) {
                return;
            }
            try {
                await navigator.clipboard.writeText(latestRouteJson);
                setStatus(routeStatus, "success", "Route JSON copied to clipboard.");
            } catch (error) {
                setStatus(routeStatus, "error", "Clipboard copy failed. Copy manually from the route panel.");
            }
        });
    }

    mediaInput.addEventListener("change", () => {
        clearResults();
        setStatus(statusMessage, "", "");
        updatePreview();
    });

    thresholdRange.addEventListener("input", updateThresholdDisplay);

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(form);

        clearResults();
        setStatus(statusMessage, "", "Running inference. This can take several seconds.");
        runButton.disabled = true;

        try {
            const response = await fetch(inferEndpoint, {
                method: "POST",
                headers: {
                    "X-CSRFToken": csrfInput ? csrfInput.value : "",
                },
                credentials: "same-origin",
                body: formData,
            });
            const raw = await response.text();
            let data = {};
            try {
                data = raw ? JSON.parse(raw) : {};
            } catch (error) {
                data = {
                    success: false,
                    error: raw || `Inference request failed with HTTP ${response.status}.`,
                };
            }
            if (!response.ok || !data.success) {
                throw new Error(data.error || `Inference request failed with HTTP ${response.status}.`);
            }

            latestResults = data.results || [];
            latestResultsJson = JSON.stringify(latestResults, null, 2);
            resultsPanel.hidden = false;
            emergencyPanel.hidden = false;
            renderRiskBanner(latestResults);
            renderResultCards(latestResults);
            setStatus(statusMessage, "success", "Inference completed successfully.");
        } catch (error) {
            const fallback = mediaInput.files && mediaInput.files.length > 0
                ? "Network error during upload/inference. If the video is large, try a smaller file or raise server upload limits."
                : "Inference request failed.";
            setStatus(statusMessage, "error", error.message || fallback);
        } finally {
            runButton.disabled = false;
        }
    });

    downloadResultsButton.addEventListener("click", () => {
        if (!latestResultsJson) {
            setStatus(statusMessage, "error", "No results available to download.");
            return;
        }

        const blob = new Blob([latestResultsJson], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "inference-results.json";
        anchor.click();
        URL.revokeObjectURL(url);
    });

    prepareEmergencyButton.addEventListener("click", async () => {
        if (latestResults.length === 0) {
            setStatus(emergencyStatus, "error", "Run inference first to prepare emergency details from model outputs.");
            return;
        }

        prepareEmergencyButton.disabled = true;
        setStatus(emergencyStatus, "", "Preparing emergency summary from inference results...");
        resetInferenceEmergencyUi();

        try {
            const data = await sendJson(emergencyEndpoint, {
                location: incidentLocation.value.trim(),
                notes: incidentNotes.value.trim(),
                results: latestResults,
            });

            setStatus(emergencyStatus, "success", `${data.message} Incident ID: ${data.incident_id}`);
            callEmergencyLink.href = `tel:${data.call_number}`;
            callEmergencyLink.textContent = `Call ${data.call_number}`;
            callEmergencyLink.hidden = false;
            copySummaryButton.hidden = false;
            incidentSummary.hidden = false;
            incidentSummary.textContent = data.summary;
        } catch (error) {
            setStatus(emergencyStatus, "error", error.message);
        } finally {
            prepareEmergencyButton.disabled = false;
        }
    });

    copySummaryButton.addEventListener("click", async () => {
        if (!incidentSummary.textContent) {
            return;
        }
        try {
            await navigator.clipboard.writeText(incidentSummary.textContent);
            setStatus(emergencyStatus, "success", "Inference emergency summary copied to clipboard.");
        } catch (error) {
            setStatus(emergencyStatus, "error", "Clipboard copy failed. Copy manually from the summary box.");
        }
    });

    manualEmergencyButton.addEventListener("click", async () => {
        manualEmergencyButton.disabled = true;
        setStatus(manualEmergencyStatus, "", "Preparing standalone emergency report...");
        manualCallLink.hidden = true;
        manualCopySummaryButton.hidden = true;
        manualIncidentSummary.hidden = true;
        manualIncidentSummary.textContent = "";

        try {
            const data = await sendJson(emergencyStandaloneEndpoint, {
                location: manualLocation.value.trim(),
                incident_type: manualIncidentType.value,
                severity: manualSeverity.value,
                notes: manualNotes.value.trim(),
            });

            setStatus(manualEmergencyStatus, "success", `${data.message} Incident ID: ${data.incident_id}`);
            manualCallLink.href = `tel:${data.call_number}`;
            manualCallLink.textContent = `Call ${data.call_number}`;
            manualCallLink.hidden = false;
            manualCopySummaryButton.hidden = false;
            manualIncidentSummary.hidden = false;
            manualIncidentSummary.textContent = data.summary;
        } catch (error) {
            setStatus(manualEmergencyStatus, "error", error.message);
        } finally {
            manualEmergencyButton.disabled = false;
        }
    });

    manualCopySummaryButton.addEventListener("click", async () => {
        if (!manualIncidentSummary.textContent) {
            return;
        }
        try {
            await navigator.clipboard.writeText(manualIncidentSummary.textContent);
            setStatus(manualEmergencyStatus, "success", "Standalone emergency summary copied to clipboard.");
        } catch (error) {
            setStatus(manualEmergencyStatus, "error", "Clipboard copy failed. Copy manually from the summary box.");
        }
    });

    faqSearch.addEventListener("input", () => {
        const query = faqSearch.value.trim().toLowerCase();
        const items = faqList.querySelectorAll("details");
        for (const item of items) {
            const text = item.textContent.toLowerCase();
            item.hidden = query && !text.includes(query);
        }
    });

    contactForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        contactSubmitButton.disabled = true;
        setStatus(contactStatus, "", "Submitting message...");

        try {
            const data = await sendJson(contactEndpoint, {
                name: contactName.value.trim(),
                email: contactEmail.value.trim(),
                topic: contactTopic.value,
                message: contactMessage.value.trim(),
            });
            setStatus(contactStatus, "success", `${data.message} Ticket: ${data.ticket_id}`);
            contactForm.reset();
        } catch (error) {
            setStatus(contactStatus, "error", error.message);
        } finally {
            contactSubmitButton.disabled = false;
        }
    });

    contactClearButton.addEventListener("click", () => {
        contactForm.reset();
        setStatus(contactStatus, "", "");
    });

    updateThresholdDisplay();
});
