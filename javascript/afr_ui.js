(function () {
    "use strict";

    var AFR_PREFIX = "afr_";

    function onAfrEnabledChange() {
        var enabledCheckbox = gradioApp().querySelector("#afr_enabled input[type=checkbox]");
        var accordionBody = gradioApp().querySelector("#afr_enabled");
        if (!enabledCheckbox || !accordionBody) return;
        var isChecked = enabledCheckbox.checked;
        var controls = accordionBody.parentElement.querySelectorAll(
            '.gr-panel > .gr-box > .gr-input, .gr-panel > .gr-box > .gr-slider, .gr-panel > .gr-box > .gr-textbox'
        );
        controls.forEach(function (el) {
            el.style.opacity = isChecked ? "1" : "0.5";
            el.style.pointerEvents = isChecked ? "auto" : "none";
        });
    }

    function setupAfrObservers() {
        var enabledCheckbox = gradioApp().querySelector("#afr_enabled input[type=checkbox]");
        if (enabledCheckbox) {
            enabledCheckbox.addEventListener("change", onAfrEnabledChange);
            onAfrEnabledChange();
        }
    }

    onUiLoaded(setupAfrObservers);
})();
