// ─── i3.cedar ──────────────────────────────────────────────────────────
//
// Cedar 4.x policy set for the Implicit Interaction Intelligence (I³)
// application-level authorization model.
//
// Entity types:
//   User, Admin, Service  (principals)
//   Diary, UserProfile, Adaptation  (resources)
//
// Actions (defined in schema.cedarschema.json):
//   Action::"read", Action::"write", Action::"export",
//   Action::"delete", Action::"bulkDelete"
//
// Context attributes consumed by these policies:
//   context.reason       : String?   // required for admin delete
//   context.sanitized    : Bool?     // true iff request came from
//                                   //   the cloud sanitizer path
//   context.mfa          : Bool?     // enforces MFA for exports
//   context.ip_trusted   : Bool?     // on-prem / VPN network flag
//

// ────────────────────────────────────────────────────────────────────
// 1. Owner-only diary access
// ────────────────────────────────────────────────────────────────────
// A user may read their own diary entries.  No other user may.
permit (
    principal,
    action == Action::"read",
    resource
)
when {
    resource has owner &&
    resource.owner == principal
};

// Owners can write their own diary.
permit (
    principal,
    action == Action::"write",
    resource
)
when {
    resource has owner &&
    resource.owner == principal
};

// ────────────────────────────────────────────────────────────────────
// 2. Export requires owner + MFA
// ────────────────────────────────────────────────────────────────────
permit (
    principal,
    action == Action::"export",
    resource
)
when {
    resource has owner &&
    resource.owner == principal &&
    context has mfa &&
    context.mfa == true
};

// ────────────────────────────────────────────────────────────────────
// 3. Owner may delete their own data
// ────────────────────────────────────────────────────────────────────
permit (
    principal,
    action == Action::"delete",
    resource
)
when {
    resource has owner &&
    resource.owner == principal
};

// ────────────────────────────────────────────────────────────────────
// 4. Admins — narrow delete powers only
// ────────────────────────────────────────────────────────────────────
// Admins can delete only with a non-empty `reason` in the context.
permit (
    principal in Admin::"admins",
    action == Action::"delete",
    resource
)
when {
    context has reason &&
    context.reason != ""
};

// Bulk delete requires a reason AND a trusted-IP session.
permit (
    principal in Admin::"admins",
    action == Action::"bulkDelete",
    resource
)
when {
    context has reason &&
    context.reason != "" &&
    context has ip_trusted &&
    context.ip_trusted == true
};

// Admins explicitly cannot read diaries.  Use `forbid` to override any
// broader permit that might be introduced later.
forbid (
    principal in Admin::"admins",
    action == Action::"read",
    resource is Diary
);

// ────────────────────────────────────────────────────────────────────
// 5. Cloud path — only sanitized state
// ────────────────────────────────────────────────────────────────────
// The cloud service principal may read an Adaptation resource only if
// the request carries `context.sanitized = true`.  Any attempt to read
// raw (un-sanitized) data from the cloud path is denied.
permit (
    principal == Service::"cloud",
    action == Action::"read",
    resource is Adaptation
)
when {
    context has sanitized &&
    context.sanitized == true
};

forbid (
    principal == Service::"cloud",
    action == Action::"read",
    resource is Adaptation
)
unless {
    context has sanitized &&
    context.sanitized == true
};

// The cloud principal may never touch a Diary.
forbid (
    principal == Service::"cloud",
    action,
    resource is Diary
);

// ────────────────────────────────────────────────────────────────────
// 6. Profile access
// ────────────────────────────────────────────────────────────────────
// Users may read and write only their own UserProfile.
permit (
    principal,
    action in [Action::"read", Action::"write"],
    resource is UserProfile
)
when {
    resource has owner &&
    resource.owner == principal
};
