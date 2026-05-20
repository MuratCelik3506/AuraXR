using UnityEngine;

public class HandProximityVisibility : MonoBehaviour
{
    [Header("References")]
    public Transform leftController;    // drag OVRLeftControllerVisual
    public Transform rightController;   // drag OVRRightControllerVisual
    public GameObject leftHandRig;
    public GameObject rightHandRig;

    [Header("Thresholds")]
    public float showDistance = 0.4f;
    public float hideDistance = 0.6f;
    public float fadeSpeed = 3f;

    private SkinnedMeshRenderer[] _leftHandRenderers;
    private SkinnedMeshRenderer[] _rightHandRenderers;
    private Renderer[] _leftControllerRenderers;
    private Renderer[] _rightControllerRenderers;
    private float _leftAlpha = 0f;
    private float _rightAlpha = 0f;
    private AuraXR.InteractableObject[] _allInteractables;

    void Start()
    {
        if (leftHandRig  == null) { Debug.LogError("[HandProximityVisibility] leftHandRig not assigned."); enabled = false; return; }
        if (rightHandRig == null) { Debug.LogError("[HandProximityVisibility] rightHandRig not assigned."); enabled = false; return; }
        if (leftController  == null) { Debug.LogError("[HandProximityVisibility] leftController not assigned."); enabled = false; return; }
        if (rightController == null) { Debug.LogError("[HandProximityVisibility] rightController not assigned."); enabled = false; return; }

        _leftHandRenderers  = leftHandRig.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        _rightHandRenderers = rightHandRig.GetComponentsInChildren<SkinnedMeshRenderer>(true);

        // Collect all renderers on the controller visual and its children automatically
        _leftControllerRenderers  = leftController.GetComponentsInChildren<Renderer>();
        _rightControllerRenderers = rightController.GetComponentsInChildren<Renderer>();

        // Find interactable objects directly — no Physics layer config required
        _allInteractables = FindObjectsByType<AuraXR.InteractableObject>(FindObjectsInactive.Exclude);
        if (_allInteractables.Length == 0)
            Debug.LogWarning("[HandProximityVisibility] No InteractableObjects found — hands will not fade in near objects. Add InteractableObject component to grabable props.");

        if (_leftHandRenderers.Length == 0)
            Debug.LogWarning("[HandProximityVisibility] leftHandRig has no SkinnedMeshRenderer children — hand will never appear.");
        if (_rightHandRenderers.Length == 0)
            Debug.LogWarning("[HandProximityVisibility] rightHandRig has no SkinnedMeshRenderer children — hand will never appear.");

        SetHandAlpha(_leftHandRenderers, 0f);
        SetHandAlpha(_rightHandRenderers, 0f);
    }

    void Update()
    {
        float leftDist  = NearestInteractableDistance(leftController.position);
        float rightDist = NearestInteractableDistance(rightController.position);

        float leftTarget  = leftDist  < showDistance ? 1f : (leftDist  > hideDistance ? 0f : _leftAlpha);
        float rightTarget = rightDist < showDistance ? 1f : (rightDist > hideDistance ? 0f : _rightAlpha);

        _leftAlpha  = Mathf.MoveTowards(_leftAlpha,  leftTarget,  fadeSpeed * Time.deltaTime);
        _rightAlpha = Mathf.MoveTowards(_rightAlpha, rightTarget, fadeSpeed * Time.deltaTime);

        SetHandAlpha(_leftHandRenderers,  _leftAlpha);
        SetHandAlpha(_rightHandRenderers, _rightAlpha);

        // Cross-fade: controller fades out as hand fades in
        SetRendererArrayAlpha(_leftControllerRenderers,  1f - _leftAlpha);
        SetRendererArrayAlpha(_rightControllerRenderers, 1f - _rightAlpha);
    }

    float NearestInteractableDistance(Vector3 origin)
    {
        float minDist = float.MaxValue;
        foreach (var io in _allInteractables)
        {
            if (io == null) continue;
            float d = Vector3.Distance(origin, io.transform.position);
            if (d < minDist) minDist = d;
        }
        return minDist;
    }

    void SetHandAlpha(SkinnedMeshRenderer[] renderers, float alpha)
    {
        foreach (var r in renderers)
        {
            Color c = r.material.color;
            c.a = alpha;
            r.material.color = c;
        }
    }

    void SetRendererArrayAlpha(Renderer[] renderers, float alpha)
    {
        foreach (var r in renderers)
        {
            Color c = r.material.color;
            c.a = alpha;
            r.material.color = c;
        }
    }
}
