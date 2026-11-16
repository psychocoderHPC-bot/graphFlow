/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <llamaLite/llamaLite.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

/** Declare one attribute tag and its field descriptor from a single name and value type.
 *
 * `GRAPHFLOW_ATTRIBUTE(Id, std::uint64_t)` expands to a tag `IdTag` derived from
 * `graphFlow::TagBase` and a field alias `IdField` equal to `graphFlow::Field<IdTag, std::uint64_t>`.
 * The tag is empty and the alias is a pure type descriptor, so the pair can be used everywhere a
 * hand-written tag and field are accepted. `T` must be trivially copyable.
 */
#define GRAPHFLOW_ATTRIBUTE(Name, T)                                                                                  \
    struct Name##Tag : ::graphFlow::TagBase                                                                           \
    {                                                                                                                 \
    };                                                                                                                \
    using Name##Field = ::graphFlow::Field<Name##Tag, T>

/** Declare one attribute tag and a `constexpr` value that names it.
 *
 * `GRAPHFLOW_TAG(Id, idTag)` expands to `struct IdTag : graphFlow::TagBase {};` and
 * `inline constexpr IdTag idTag{};`, so `record[idTag]` reads or writes the attribute. The one-argument
 * form `GRAPHFLOW_TAG(Id)` names the value `IdTag_v`. Both forms keep the tag empty, so passing a tag
 * value to `operator[]` carries no runtime state and adds no record layout or performance cost.
 *
 * This macro is additive: `GRAPHFLOW_ATTRIBUTE` and hand-written `struct Name##Tag : TagBase {};`
 * declarations keep working unchanged. Use this macro when a named value makes the call site clearer
 * than `record.get<IdTag>()`; declare the matching `Field` aliases next to it.
 */
#define GRAPHFLOW_TAG_ONE(Name)                                                                                       \
    struct Name##Tag : ::graphFlow::TagBase                                                                           \
    {                                                                                                                 \
    };                                                                                                                \
    inline constexpr Name##Tag Name##Tag##_v                                                                          \
    {                                                                                                                 \
    }

#define GRAPHFLOW_TAG_NAMED(Name, Variable)                                                                           \
    struct Name##Tag : ::graphFlow::TagBase                                                                           \
    {                                                                                                                 \
    };                                                                                                                \
    inline constexpr Name##Tag Variable                                                                               \
    {                                                                                                                 \
    }

#define GRAPHFLOW_TAG_SELECT(_1, _2, Macro, ...) Macro

#define GRAPHFLOW_TAG(...) GRAPHFLOW_TAG_SELECT(__VA_ARGS__, GRAPHFLOW_TAG_NAMED, GRAPHFLOW_TAG_ONE)(__VA_ARGS__)

namespace graphFlow
{
    /** Base class of an attribute tag, re-exported from llamaLite.
     *
     * A tag is an empty type derived from `graphFlow::TagBase`. It selects the attribute and carries
     * no runtime data. Two attributes with the same tag cannot coexist in one record.
     */
    using TagBase = llama_lite::TagBase;

    /** Compile-time named attribute of a record, backed by `llama_lite::Field`.
     *
     * `Tag` must derive from `TagBase` and be empty. `T` is the stored value type and must be
     * trivially copyable so a record remains copyable between host and device memory. The field is
     * a pure type descriptor; values live in `Record` or in the columns of `SoAStorage`.
     */
    template<llama_lite::IsTag Tag, typename T>
    using Field = llama_lite::Field<Tag, T>;

    namespace detail
    {
        template<typename T>
        inline constexpr std::uint64_t paddedAttributeBytes
            = (static_cast<std::uint64_t>(sizeof(T)) + static_cast<std::uint64_t>(alignof(T)) - 1u)
              / static_cast<std::uint64_t>(alignof(T)) * static_cast<std::uint64_t>(alignof(T));

    } // namespace detail

    /** Set of attribute tags a node is allowed to read or write, backed by `llama_lite::Set`.
     *
     * A node declares the narrowest set it needs. `RecordProxy` rejects a `get<Tag>()` or
     * `set<Tag>()` whose tag is not in the set at compile time, while the underlying record still
     * keeps every attribute.
     */
    template<typename... Tags>
    struct Set : llama_lite::Set<Tags...>
    {
        static constexpr std::size_t size = sizeof...(Tags);

        template<typename Tag>
        static constexpr bool contains = (std::same_as<Tag, Tags> || ...);
    };

    /** Validated, ordered attribute list backed by `llama_lite::Record`.
     *
     * The list is the single source of truth for the attribute names, order, and value types of a
     * `Record` or `SoAStorage`. `llama_lite::Record` validates that the fields are `llama_lite::Field`
     * descriptors and that their tags are unique; the adapter adds the trivially-copyable value
     * constraint and the slot accounting.
     *
     * `slotBytes` is the accounting size of one record in an AoS slot: the sum of the naturally
     * aligned attribute widths plus a 32-bit scratch word used by the scatter path for bin ids.
     * Each attribute has its own column allocation, so the slot bytes are a budget and view size,
     * not one contiguous array.
     */
    template<typename... Fields>
    struct AttributeList
    {
        static_assert(sizeof...(Fields) > 0u, "graphFlow: an attribute list must contain at least one field");
        static_assert(
            (llama_lite::IsField<Fields> && ...),
            "graphFlow: every attribute must be a graphFlow::Field<Tag, T>");
        static_assert(
            (std::is_trivially_copyable_v<typename Fields::value_type> && ...),
            "graphFlow: attribute value types must be trivially copyable");

        /** llamaLite record descriptor that owns tag validation and path lookup. */
        using record_type = llama_lite::Record<Fields...>;
        using field_tuple = std::tuple<Fields...>;
        using attribute_set = Set<typename Fields::tag_type...>;

        // Force the descriptor's instantiation so its duplicate-tag validation also fires when an
        // AttributeList is named as a type but never turned into a record or storage.
        static_assert(sizeof(record_type) > 0u);

        static constexpr std::size_t size = sizeof...(Fields);
        static constexpr std::uint64_t scratchBytes = sizeof(std::uint32_t);
        static constexpr std::uint64_t attributeBytes
            = (detail::paddedAttributeBytes<typename Fields::value_type> + ... + 0u);
        static constexpr std::uint64_t slotBytes = attributeBytes + scratchBytes;

        template<typename Tag>
        static constexpr bool contains = (std::same_as<Tag, typename Fields::tag_type> || ...);

        template<typename Tag>
        static constexpr std::size_t index = llama_lite::indexOfType<Tag, typename Fields::tag_type...>();

        template<typename Tag>
        using field_type = std::tuple_element_t<index<Tag>, field_tuple>;

        template<typename Tag>
        using value_type = typename field_type<Tag>::value_type;
    };

    /** Short spelling of `AttributeList` for field packs gathered from several attributes.
     *
     * `Attributes<IdField, WeightField>` is the same type as
     * `AttributeList<IdField, WeightField>`; the alias keeps a `Record` and its storage declaration
     * readable when the fields come from `GRAPHFLOW_ATTRIBUTE`.
     */
    template<typename... Fields>
    using Attributes = AttributeList<Fields...>;

    namespace detail
    {
        template<typename TSet, typename TAttributeList>
        struct SetCompatible;

        template<typename... Tags, typename... Fields>
        struct SetCompatible<Set<Tags...>, AttributeList<Fields...>>
            : std::bool_constant<((AttributeList<Fields...>::template contains<Tags>) && ...)>
        {
        };

        /** Proxy access to a whole record kept in AoS memory. */
        template<typename TRecord>
        struct RefRecordAccess
        {
            using record_type = TRecord;

            TRecord* record = nullptr;

            template<typename Tag>
            ALPAKA_FN_ACC constexpr decltype(auto) get() const
            {
                return record->template get<Tag>();
            }

            template<typename Tag, typename U>
            ALPAKA_FN_ACC constexpr void set(U&& value)
            {
                record->template set<Tag>(std::forward<U>(value));
            }

            ALPAKA_FN_ACC constexpr record_type load() const
            {
                return *record;
            }

            ALPAKA_FN_ACC constexpr void store(record_type const& value)
            {
                *record = value;
            }
        };

        /** Read-only proxy access to a whole record kept in AoS memory. */
        template<typename TRecord>
        struct ConstRefRecordAccess
        {
            using record_type = TRecord;

            TRecord const* record = nullptr;

            template<typename Tag>
            ALPAKA_FN_ACC constexpr decltype(auto) get() const
            {
                return record->template get<Tag>();
            }

            ALPAKA_FN_ACC constexpr record_type load() const
            {
                return *record;
            }
        };
    } // namespace detail

    /** Narrow, per-node view of a record.
     *
     * A node receives a proxy constrained to its `Set<Tags...>`. `get<Tag>()` and `set<Tag>()`
     * outside that set fail at compile time. Copying the proxy to a record copies every attribute,
     * including the ones the node is not allowed to touch, so a full record survives a stage that
     * only declares a subset.
     */
    template<typename TAccess, typename TSet>
    class RecordProxy
    {
        static_assert(
            detail::SetCompatible<TSet, typename TAccess::record_type::attribute_list>::value,
            "graphFlow::RecordProxy: Set contains a tag that is not an attribute of the record");

    public:
        using access_type = TAccess;
        using set_type = TSet;
        using record_type = typename TAccess::record_type;

        constexpr RecordProxy() = default;

        ALPAKA_FN_ACC constexpr explicit RecordProxy(access_type access) : m_access{access}
        {
        }

        template<typename Tag>
        ALPAKA_FN_ACC constexpr decltype(auto) get() const
        {
            static_assert(
                TSet::template contains<Tag>,
                "graphFlow::RecordProxy: get<Tag>() is outside the node's attribute subset");
            return m_access.template get<Tag>();
        }

        template<typename Tag, typename U>
        ALPAKA_FN_ACC constexpr void set(U&& value)
        {
            static_assert(
                TSet::template contains<Tag>,
                "graphFlow::RecordProxy: set<Tag>() is outside the node's attribute subset");
            m_access.template set<Tag>(std::forward<U>(value));
        }

        /** Subscript spelling of get<Tag>(), constrained to the proxy set. */
        template<typename Tag>
        [[nodiscard]] ALPAKA_FN_ACC constexpr decltype(auto) operator[](Tag)
        {
            return get<Tag>();
        }

        /** Subscript spelling of get<Tag>() for a const proxy or a const record. */
        template<typename Tag>
        [[nodiscard]] ALPAKA_FN_ACC constexpr decltype(auto) operator[](Tag) const
        {
            return get<Tag>();
        }

        [[nodiscard]] ALPAKA_FN_ACC constexpr record_type load() const
        {
            return m_access.load();
        }

        ALPAKA_FN_ACC constexpr void store(record_type const& record)
        {
            m_access.store(record);
        }

        ALPAKA_FN_ACC constexpr operator record_type() const
        {
            return load();
        }

    private:
        access_type m_access{};
    };

    /** Mutable proxy type of a record value, for functors that name the parameter type.
     *
     * `flow.transform<Set<Tags...>>(...)` hands its functor this proxy over a local record copy.
     */
    template<typename TRecord, typename TSet>
    using MutableRecordProxy = RecordProxy<detail::RefRecordAccess<TRecord>, TSet>;

    /** Read-only proxy type of a record value. */
    template<typename TRecord, typename TSet>
    using ConstRecordProxy = RecordProxy<detail::ConstRefRecordAccess<TRecord>, TSet>;

    /** Value record with named attributes and value semantics.
     *
     * The value storage is a one-element `llama_lite::SoA` over the `llama_lite::Record` descriptor,
     * so copying a `Record` copies every attribute and the layout is still a set of per-attribute
     * arrays. Use `proxy<Set<Tags...>>()` to hand a node a narrow view without changing the record
     * layout.
     */
    template<typename... Fields>
    class Record
    {
        static_assert(sizeof...(Fields) > 0u, "graphFlow: a record must contain at least one field");

        using descriptor_type = llama_lite::Record<Fields...>;
        using storage_type = llama_lite::SoA<descriptor_type, 1u, llama_lite::CompactAlignment>;

    public:
        using attribute_list = AttributeList<Fields...>;
        using attribute_set = typename attribute_list::attribute_set;
        using record_type = descriptor_type;

        static constexpr std::size_t size = sizeof...(Fields);

        constexpr Record() = default;

        /** Construct a record from one value per attribute, in declaration order. */
        template<typename... Values>
        requires(sizeof...(Values) == sizeof...(Fields))
        constexpr explicit Record(Values&&... values)
        {
            assign(std::index_sequence_for<Fields...>{}, std::forward<Values>(values)...);
        }

        // getLeaf() deduces its tag argument, so `m_values.template getLeaf(...)` is not a
        // simple-template-id; clang rejects the disambiguator as ill-formed.
        template<typename Tag>
        [[nodiscard]] ALPAKA_FN_ACC constexpr decltype(auto) get()
        {
            static_assert(
                attribute_list::template contains<Tag>,
                "graphFlow::Record: get<Tag>() names an unknown attribute");
            return m_values.getLeaf(Tag{})[0u];
        }

        template<typename Tag>
        [[nodiscard]] ALPAKA_FN_ACC constexpr decltype(auto) get() const
        {
            static_assert(
                attribute_list::template contains<Tag>,
                "graphFlow::Record: get<Tag>() names an unknown attribute");
            return m_values.getLeaf(Tag{})[0u];
        }

        template<typename Tag, typename U>
        ALPAKA_FN_ACC constexpr void set(U&& value)
        {
            static_assert(
                attribute_list::template contains<Tag>,
                "graphFlow::Record: set<Tag>() names an unknown attribute");
            m_values.getLeaf(Tag{})[0u] = std::forward<U>(value);
        }

        /** Subscript spelling of get<Tag>() and set<Tag>(). */
        template<typename Tag>
        [[nodiscard]] ALPAKA_FN_ACC constexpr decltype(auto) operator[](Tag)
        {
            return get<Tag>();
        }

        /** Subscript spelling of get<Tag>() on a const record. */
        template<typename Tag>
        [[nodiscard]] ALPAKA_FN_ACC constexpr decltype(auto) operator[](Tag) const
        {
            return get<Tag>();
        }

        template<typename TSet = attribute_set>
        [[nodiscard]] ALPAKA_FN_ACC auto proxy() -> RecordProxy<detail::RefRecordAccess<Record<Fields...>>, TSet>;

        template<typename TSet = attribute_set>
        [[nodiscard]] ALPAKA_FN_ACC auto proxy() const
            -> RecordProxy<detail::ConstRefRecordAccess<Record<Fields...>>, TSet>;

    private:
        template<std::size_t... Is, typename... Values>
        constexpr void assign(std::index_sequence<Is...>, Values&&... values)
        {
            (set<typename std::tuple_element_t<Is, std::tuple<Fields...>>::tag_type>(std::forward<Values>(values)),
             ...);
        }

        storage_type m_values{};
    };

    template<typename... Fields>
    template<typename TSet>
    ALPAKA_FN_ACC auto Record<Fields...>::proxy() -> RecordProxy<detail::RefRecordAccess<Record<Fields...>>, TSet>
    {
        static_assert(
            detail::SetCompatible<TSet, attribute_list>::value,
            "graphFlow::Record: proxy Set contains a tag that is not an attribute of the record");
        return RecordProxy<detail::RefRecordAccess<Record<Fields...>>, TSet>{
            detail::RefRecordAccess<Record<Fields...>>{this}};
    }

    template<typename... Fields>
    template<typename TSet>
    ALPAKA_FN_ACC auto Record<Fields...>::proxy() const
        -> RecordProxy<detail::ConstRefRecordAccess<Record<Fields...>>, TSet>
    {
        static_assert(
            detail::SetCompatible<TSet, attribute_list>::value,
            "graphFlow::Record: proxy Set contains a tag that is not an attribute of the record");
        return RecordProxy<detail::ConstRefRecordAccess<Record<Fields...>>, TSet>{
            detail::ConstRefRecordAccess<Record<Fields...>>{this}};
    }
} // namespace graphFlow
